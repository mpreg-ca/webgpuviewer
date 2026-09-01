package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blitCachedRegion
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * A dual-page book turn: one leaf lifts off the spine, curls, and lands on the other side.
 *
 * Forward, the leaf shows page 1's right half in front and page 2's left behind, each at its own
 * size; a side with no page is blank in the background colour. The halves that stay put come from
 * the caches, clipped at their spine - see [blitCachedRegion].
 *
 * Geometry is in width fractions from the surface centre, y over the aspect ratio. No depth
 * attachment: height rises with the tangent angle while it stays inside PI, so strips emitted
 * spine-outwards land back to front.
 */
object TransitionDualFlip : Transition() {
    override val premultipliedOutput = true

    private const val UNIFORM_SIZE = 96

    /** Total curl at the halfway point, in radians of tangent turn from spine to outer edge. */
    private const val BEND = 0.95f

    /** Never zero - the arc's radius is length/bend. */
    private const val MIN_BEND = 0.04f

    // Along the leaf only - it does not bend vertically, so rows buy just a shorter diagonal.
    private const val COLS = 64
    private const val ROWS = 2
    private const val SHEET_VERTICES = COLS * ROWS * 6

    /** Shadow grid then leaf grid, in one draw - see `vs_main`. */
    private const val VERTICES = SHEET_VERTICES * 2

    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(UNIFORM_SIZE).order(ByteOrder.nativeOrder())
    }

    private val flipSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(magFilter = FilterMode.Linear, minFilter = FilterMode.Linear)
        )
    }

    /** The turning leaf for one frame. */
    private class Leaf(
        /** The face it starts on, in page 1's cache: normalised (x1, y1, x2, y2). */
        val frontRect: FloatArray,
        /** The face it lands on, in page 2's cache. Mirrors [frontRect] when there is none. */
        val backRect: FloatArray,
        val spine: Float,
        /** +1 resting right of the spine, -1 left. */
        val dir: Float,
        /** Rotation about the spine, 0 (at rest) to PI (landed). */
        val phi: Float,
        /** Tangent turn from spine to outer edge - the curl. [phi] + this <= PI. */
        val bend: Float,
        /** Spine to the far edge of the sheet - both faces fit inside it. */
        val len: Float,
        val top: Float,
        val bottom: Float,
        val aspect: Float,
        /** Whether that face has a cached page to sample; blank if not. */
        val hasFront: Boolean,
        val hasBack: Boolean,
    )

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)
        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        // Clamped: a fling can carry the offset past a page, and a negative bend inverts the arc.
        val t = abs(frac).coerceIn(0f, 1f)
        // frac > 0 brings page 2 in from the right, as TransitionBasic does: the right leaf turns left.
        val forward = frac > 0f
        val spine1 = page1.spineX(dst)
        val spine2 = page2.spineX(dst)

        val background = blendBackgroundColor(
            page1.backgroundColor ?: 0xFF000000.toInt(),
            page2.backgroundColor ?: 0xFF000000.toInt(),
            t,
        )

        val pass = beginClearedPass(encoder, dst)
        try {
            Draw.rect(pass, 0f, 0f, 1f, 1f, background)
            // Clipped at each spine: page 1 keeps the side the leaf left, page 2 the one it uncovers.
            if (forward) {
                spine1?.let { blitCachedRegion(pass, cached1, 0f, 0f, it, 1f) }
                spine2?.let { blitCachedRegion(pass, cached2, it, 0f, 1f, 1f) }
            } else {
                spine1?.let { blitCachedRegion(pass, cached1, it, 0f, 1f, 1f) }
                spine2?.let { blitCachedRegion(pass, cached2, 0f, 0f, it, 1f) }
            }
        } finally {
            pass.end()
        }

        val leaf = leaf(page1, page2, dst, t, forward, spine1, spine2, cached1, cached2) ?: return
        // A blank face is never sampled, so the surviving view stands in - [leaf] rules out both.
        val front = cached1 ?: cached2 ?: return
        val back = cached2 ?: cached1 ?: return

        val leafPass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Load,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )
        try {
            bind(leafPass, leaf, front, back, background)
            leafPass.draw(VERTICES)
        } finally {
            leafPass.end()
        }
    }

    /**
     * The leaf at [t], or null when there is nothing to turn. The sheet takes in both faces and
     * holds that size throughout, each drawn at its own size on it - so neither resizes into the
     * other, and neither snaps at its own end.
     */
    private fun leaf(
        page1: ImagePage,
        page2: ImagePage,
        dst: GPUTexture,
        t: Float,
        forward: Boolean,
        spine1: Float?,
        spine2: Float?,
        cached1: GPUTextureView?,
        cached2: GPUTextureView?,
    ): Leaf? {
        // Page 1's hinge: blending toward page 2's would slide the fold across mid-turn.
        val spine = spine1 ?: spine2 ?: return null
        // Forward: front is page 1's right half, back is page 2's left. Backward mirrors both.
        val dir = if (forward) 1f else -1f
        val rawFront = page1.leafRect(dst, left = !forward)
        val rawBack = page2.leafRect(dst, left = forward)
        // A side with no page mirrors the other across the spine, to size its blank sheet by.
        val frontRect = rawFront ?: rawBack?.let { mirror(it, spine) } ?: return null
        val backRect = rawBack ?: mirror(frontRect, spine)

        val lenFront = if (forward) frontRect[2] - spine else spine - frontRect[0]
        val lenBack = if (forward) spine - backRect[0] else backRect[2] - spine
        val len = max(lenFront, lenBack)
        if (len <= 0f) return null

        // Flat at both ends, curliest halfway.
        val bend = MIN_BEND + (BEND - MIN_BEND) * sin(PI * t).toFloat()

        return Leaf(
            frontRect = frontRect,
            backRect = backRect,
            spine = spine,
            dir = dir,
            // Held back so the outer edge stops at PI - strip order needs height monotonic.
            phi = min((PI * t).toFloat(), PI.toFloat() - bend),
            bend = bend,
            len = len,
            top = min(frontRect[1], backRect[1]),
            bottom = max(frontRect[3], backRect[3]),
            aspect = dst.width.toFloat() / dst.height,
            hasFront = rawFront != null && cached1 != null,
            hasBack = rawBack != null && cached2 != null,
        )
    }

    /** [rect] reflected across the spine - where the leaf's other face has to lie. */
    private fun mirror(rect: FloatArray, spine: Float) =
        floatArrayOf(2f * spine - rect[2], rect[1], 2f * spine - rect[0], rect[3])

    /** Set [pipeline] and this frame's uniforms on [pass]. [blank] paints a face with no page. */
    private fun bind(
        pass: GPURenderPassEncoder,
        leaf: Leaf,
        front: GPUTextureView,
        back: GPUTextureView,
        blank: Int,
    ) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        for (v in leaf.frontRect) byteBuffer.putFloat(v)
        for (v in leaf.backRect) byteBuffer.putFloat(v)
        byteBuffer.putFloat(leaf.spine)
        byteBuffer.putFloat(leaf.dir)
        byteBuffer.putFloat(leaf.phi)
        byteBuffer.putFloat(leaf.bend)
        byteBuffer.putFloat(leaf.len)
        byteBuffer.putFloat(leaf.top)
        byteBuffer.putFloat(leaf.bottom)
        byteBuffer.putFloat(leaf.aspect)
        byteBuffer.putFloat(if (leaf.hasFront) 1f else 0f)
        byteBuffer.putFloat(if (leaf.hasBack) 1f else 0f)
        byteBuffer.putFloat(0f)
        byteBuffer.putFloat(0f)
        // Opaque, so premultiplied and straight agree.
        byteBuffer.putFloat(((blank shr 16) and 0xFF) / 255f)
        byteBuffer.putFloat(((blank shr 8) and 0xFF) / 255f)
        byteBuffer.putFloat((blank and 0xFF) / 255f)
        byteBuffer.putFloat(1f)
        byteBuffer.flip()

        val uniformBuffer = device.createBuffer(
            GPUBufferDescriptor(
                size = UNIFORM_SIZE.toLong(), usage = BufferUsage.Uniform or BufferUsage.CopyDst
            )
        )
        device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = uniformBuffer),
                        GPUBindGroupEntry(1, textureView = front),
                        GPUBindGroupEntry(2, textureView = back),
                        GPUBindGroupEntry(3, sampler = flipSampler),
                    )
                )
            )
        )
    }

    /**
     * The cross-section is a circular arc of radius `len / bend` hinged on the spine, its tangent
     * running `phi` to `phi + bend`, so the sheet swings and curls at once. Heights are width
     * fractions like x, so the arc is round rather than stretched.
     */
    override val code = """
struct Uniforms {
    // The leaf's two faces within their cached surfaces: normalised (x1, y1, x2, y2).
    front_rect: vec4<f32>,
    back_rect: vec4<f32>,
    // spine x, direction (+1 resting right of the spine), turn angle, curl
    geom: vec4<f32>,
    // sheet length, top y, bottom y, surface aspect
    span: vec4<f32>,
    // front textured, back textured, unused, unused
    flags: vec4<f32>,
    // What a face with no page behind it is painted, premultiplied.
    blank: vec4<f32>,
}

@group(0) @binding(0) var<uniform> flip: Uniforms;
@group(0) @binding(1) var front_tex: texture_2d<f32>;
@group(0) @binding(2) var back_tex: texture_2d<f32>;
@group(0) @binding(3) var flip_sampler: sampler;

const PI: f32 = 3.14159265;

// Eye distance, in the width fractions the geometry is measured in.
const EYE: f32 = 2.6;

// Key light height over the spine, in leaf lengths. Under EYE, or shadows hide under their caster.
const LIGHT_HEIGHT: f32 = 2.8;

// How fast a shadow fades as the leaf rises off the page, per leaf length.
const SHADOW_FALLOFF: f32 = 2.6;
const SHADOW_DEPTH: f32 = 0.6;

// Penumbra width, as a fraction of the face along the sheet and down it.
const SOFT_ALONG: f32 = 0.06;
const SOFT_DOWN: f32 = 0.04;

fn leaf_radius() -> f32 { return flip.span.x / flip.geom.w; }

/// The tangent angle at arc fraction [s] along the leaf - 0 at the spine, 1 at the outer edge.
fn leaf_angle(s: f32) -> f32 { return flip.geom.z + flip.geom.w * s; }

/// A point on the leaf in centred width fractions, at tangent angle [b] and vertical fraction [v].
fn leaf_point(b: f32, v: f32) -> vec3<f32> {
    let r = leaf_radius();
    let phi = flip.geom.z;
    let y = mix(flip.span.y, flip.span.z, v);
    return vec3<f32>(
        (flip.geom.x - 0.5) + flip.geom.y * r * (sin(b) - sin(phi)),
        (y - 0.5) / flip.span.w,
        r * (cos(phi) - cos(b)),
    );
}

/// Centred width fractions back to normalised surface coordinates, under perspective.
fn project(p: vec3<f32>) -> vec2<f32> {
    let s = EYE / (EYE - p.z);
    return vec2<f32>(0.5 + p.x * s, 0.5 + p.y * s * flip.span.w);
}

/// A lamp on the spine, over the middle of the book. Point, not directional, so shadows run outward.
fn light_pos() -> vec3<f32> {
    let mid_y = (0.5 * (flip.span.y + flip.span.z) - 0.5) / flip.span.w;
    return vec3<f32>(flip.geom.x - 0.5, mid_y, LIGHT_HEIGHT * flip.span.x);
}

/// Where a point on the leaf lays its shadow on the page. The clamp guards only a leaf as high as its light.
fn shadow_cast(p: vec3<f32>) -> vec2<f32> {
    let light = light_pos();
    let t = light.z / max(light.z - p.z, 0.25 * light.z);
    return light.xy + t * (p.xy - light.xy);
}

/// How dark the shadow is where the point casting it sits [z] above the page.
fn shadow_alpha(z: f32) -> f32 {
    return SHADOW_DEPTH * exp(-SHADOW_FALLOFF * max(z, 0.0) / flip.span.x);
}

/// Lambert shading at [p], tangent angle [b]. The normal has no y - the sheet bends only about
/// the spine - but the light does, so take the direction to it in full.
fn leaf_shade(p: vec3<f32>, b: f32, front: bool) -> f32 {
    var n = vec3<f32>(-flip.geom.y * sin(b), 0.0, cos(b));
    if (!front) { n = -n; }
    return 0.42 + 0.58 * max(dot(n, normalize(light_pos() - p)), 0.0);
}

/// The face showing at a point of the sheet - whichever way the surface is turned.
struct Face {
    front: bool,
    /// False past this face's edges, and in the gutter before it starts.
    covers: bool,
    /// False for a side with no page: it draws blank rather than sampling.
    textured: bool,
    /// Surface coordinate, and so the texture coordinate - which lays the face on the sheet at its
    /// own size instead of stretching it over the whole sheet.
    uv: vec2<f32>,
    rect: vec4<f32>,
    /// Which way this face runs from the spine.
    side: f32,
}

fn face_at(s: f32, v: f32, b: f32) -> Face {
    var f: Face;
    f.front = cos(b) > 0.0;
    f.side = select(-flip.geom.y, flip.geom.y, f.front);
    f.uv = vec2<f32>(flip.geom.x + f.side * s * flip.span.x, mix(flip.span.y, flip.span.z, v));
    f.rect = select(flip.back_rect, flip.front_rect, f.front);
    f.textured = select(flip.flags.y, flip.flags.x, f.front) > 0.5;
    f.covers = f.uv.x >= f.rect.x && f.uv.x <= f.rect.z &&
        f.uv.y >= f.rect.y && f.uv.y <= f.rect.w;
    return f;
}

/// Fades the shadow towards the edges the leaf lifts from - a stand-in penumbra, never at the spine.
fn shadow_softness(f: Face) -> f32 {
    let outer = select(f.rect.x, f.rect.z, f.side > 0.0);
    let soft_x = SOFT_ALONG * flip.span.x;
    let soft_y = SOFT_DOWN * max(flip.span.z - flip.span.y, 1e-5);
    return smoothstep(0.0, soft_x, abs(outer - f.uv.x)) *
        smoothstep(0.0, soft_y, f.uv.y - f.rect.y) *
        smoothstep(0.0, soft_y, f.rect.w - f.uv.y);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Arc fraction along the leaf and vertical fraction down it.
    @location(0) s: f32,
    @location(1) v: f32,
    // The sheet point, carried so the fragment stage need not redo the arc's trig.
    @location(2) world: vec3<f32>,
    // 1 on the shadow half; every vertex of a quad shares it. Not flat - compat mode rejects that.
    @location(3) shadow: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const COLS: u32 = ${COLS}u;
    const ROWS: u32 = ${ROWS}u;
    const SHEET: u32 = ${SHEET_VERTICES}u;

    // Shadow grid first, leaf over it. Split by index, not two draws - one uniform buffer, not two.
    let shadow = vertex_index < SHEET;
    let i = select(vertex_index - SHEET, vertex_index, shadow);

    let quad_index = i / 6u;
    let vert_in_quad = i % 6u;
    let col = quad_index % COLS;
    let row = quad_index / COLS;

    let s0 = f32(col) / f32(COLS);
    let s1 = f32(col + 1u) / f32(COLS);
    let v0 = f32(row) / f32(ROWS);
    let v1 = f32(row + 1u) / f32(ROWS);

    var sv: vec2<f32>;
    switch (vert_in_quad) {
        case 0u: { sv = vec2<f32>(s0, v0); }
        case 1u: { sv = vec2<f32>(s0, v1); }
        case 2u: { sv = vec2<f32>(s1, v0); }
        case 3u: { sv = vec2<f32>(s1, v0); }
        case 4u: { sv = vec2<f32>(s0, v1); }
        default: { sv = vec2<f32>(s1, v1); }
    }

    let world = leaf_point(leaf_angle(sv.x), sv.y);

    var screen: vec2<f32>;
    if (shadow) {
        // Dropped through the light onto the page, landing at z = 0 - no perspective divide.
        let flat = shadow_cast(world);
        screen = vec2<f32>(0.5 + flat.x, 0.5 + flat.y * flip.span.w);
    } else {
        screen = project(world);
    }

    var out: VertexOutput;
    out.position = vec4<f32>(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0, 0.0, 1.0);
    out.s = sv.x;
    out.v = sv.y;
    out.world = world;
    out.shadow = select(0.0, 1.0, shadow);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let b = leaf_angle(in.s);
    let face = face_at(in.s, in.v, b);

    if (!face.covers) { discard; }

    // Premultiplied black, so the shadow multiplies what is under it down rather than tinting it.
    if (in.shadow > 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, shadow_alpha(in.world.z) * shadow_softness(face));
    }

    var texel = flip.blank;
    if (face.textured) {
        if (face.front) {
            texel = textureSampleLevel(front_tex, flip_sampler, face.uv, 0.0);
        } else {
            texel = textureSampleLevel(back_tex, flip_sampler, face.uv, 0.0);
        }
    }

    // Premultiplied throughout - see premultipliedOutput - so shading scales rgb alone.
    return vec4<f32>(texel.rgb * leaf_shade(in.world, b, face.front), texel.a);
}"""
}
