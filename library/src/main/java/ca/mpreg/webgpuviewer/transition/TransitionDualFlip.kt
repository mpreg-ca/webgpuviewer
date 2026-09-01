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
import kotlin.math.sin

/**
 * A dual-page book turn: one leaf lifts off the spine, curls, and lands on the other side.
 *
 * Turning forward the leaf carries page 1's right half in front and page 2's left half behind; a
 * missing side is black, mirroring the face opposite. [ImagePage.leafRect] splits a single page
 * down the middle so it turns like a spread. The halves that stay put come straight from the
 * caches, clipped at their own spine - see [blitCachedRegion]. The leaf is a tessellated grid
 * curled in the vertex stage; its shadow is the same grid dropped onto the page through the light,
 * drawn first so the leaf covers what lies under it.
 *
 * Geometry is in width fractions from the centre of the surface, y divided by the aspect ratio to
 * share x's metric. The light hangs over the spine, so shadows run outward from it.
 *
 * No depth attachment: a curl overlaps itself, but height above the page rises with the tangent
 * angle while that stays inside PI, so emitting strips spine-outwards puts them back to front.
 */
object TransitionDualFlip : Transition() {
    override val premultipliedOutput = true

    private const val UNIFORM_SIZE = 80

    /** Total curl at the halfway point, in radians of tangent turn from spine to outer edge. */
    private const val BEND = 0.95f

    /** Never zero - the arc's radius is length/bend. */
    private const val MIN_BEND = 0.04f

    // Along the leaf only - it does not bend vertically, so rows buy just a shorter diagonal.
    private const val COLS = 64
    private const val ROWS = 2
    private const val VERTICES = COLS * ROWS * 6

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
        /** The face it lands on, in page 2's cache. Mirrors [frontRect] when there is no page. */
        val backRect: FloatArray,
        val spine: Float,
        /** +1 resting right of the spine, -1 left. */
        val dir: Float,
        /** Rotation about the spine, 0 (at rest) to PI (landed). */
        val phi: Float,
        /** Tangent turn from spine to outer edge - the curl. [phi] + this <= PI. */
        val bend: Float,
        /** Spine to outer edge, in width fractions. */
        val len: Float,
        val top: Float,
        val bottom: Float,
        val aspect: Float,
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

        val t = abs(frac)
        // frac > 0 brings page 2 in from the right, as TransitionBasic does: the right leaf turns left.
        val forward = frac > 0f
        val spine1 = page1.spineX(dst)
        val spine2 = page2.spineX(dst)

        val bg1 = page1.backgroundColor ?: 0xFF000000.toInt()
        val bg2 = page2.backgroundColor ?: 0xFF000000.toInt()

        val pass = beginClearedPass(encoder, dst)
        try {
            Draw.rect(pass, 0f, 0f, 1f, 1f, blendBackgroundColor(bg1, bg2, t))
            // Page 1 keeps the side the leaf is not on, page 2 the side it uncovers - each clipped
            // at its own spine, so neither cache brings its other half.
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

        val leaf = leaf(page1, page2, dst, t, forward, spine1, spine2) ?: return
        // A missing face is never sampled, so the other view stands in - no placeholder needed.
        val front = (if (leaf.hasFront) cached1 else cached2) ?: return
        val back = (if (leaf.hasBack) cached2 else cached1) ?: return

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
            bind(leafPass, leaf, 1f, front, back)
            leafPass.draw(VERTICES)
            bind(leafPass, leaf, 0f, front, back)
            leafPass.draw(VERTICES)
        } finally {
            leafPass.end()
        }
    }

    /**
     * The leaf at [t], or null when neither page has a half to turn. Its length and vertical extent
     * cross-fade front face to back, so differently sized pages each land on their own rect exactly;
     * both faces still sample their full rect throughout.
     */
    private fun leaf(
        page1: ImagePage,
        page2: ImagePage,
        dst: GPUTexture,
        t: Float,
        forward: Boolean,
        spine1: Float?,
        spine2: Float?,
    ): Leaf? {
        val spine = when {
            spine1 != null && spine2 != null -> spine1 + (spine2 - spine1) * t
            else -> spine1 ?: spine2 ?: return null
        }
        // Forward: front is page 1's right half, back is page 2's left. Backward mirrors both.
        val dir = if (forward) 1f else -1f
        val rawFront = page1.leafRect(dst, left = !forward)
        val rawBack = page2.leafRect(dst, left = forward)
        // No page on that side: a black leaf mirroring the other face across the spine.
        val frontRect = rawFront ?: rawBack?.let { mirror(it, spine) } ?: return null
        val backRect = rawBack ?: mirror(frontRect, spine)

        val lenFront = if (forward) frontRect[2] - spine else spine - frontRect[0]
        val lenBack = if (forward) spine - backRect[0] else backRect[2] - spine
        val len = lenFront + (lenBack - lenFront) * t
        if (len <= 0f) return null

        // Flat at both ends, curliest halfway.
        val bend = MIN_BEND + (BEND - MIN_BEND) * sin(PI * t).toFloat()

        return Leaf(
            frontRect = frontRect,
            backRect = backRect,
            spine = spine,
            dir = dir,
            // Held back so the outer edge stops at PI, keeping height monotonic along the leaf -
            // the strip ordering rests on that. Only binds in the last few percent, already flat.
            phi = minOf((PI * t).toFloat(), PI.toFloat() - bend),
            bend = bend,
            len = len,
            top = frontRect[1] + (backRect[1] - frontRect[1]) * t,
            bottom = frontRect[3] + (backRect[3] - frontRect[3]) * t,
            aspect = dst.width.toFloat() / dst.height,
            hasFront = rawFront != null,
            hasBack = rawBack != null,
        )
    }

    /** [rect] reflected across the spine - where the leaf's other face has to lie. */
    private fun mirror(rect: FloatArray, spine: Float) =
        floatArrayOf(2f * spine - rect[2], rect[1], 2f * spine - rect[0], rect[3])

    /** Set [pipeline] and this draw's uniforms on [pass]. [mode] is 0 for the leaf, 1 its shadow. */
    private fun bind(
        pass: GPURenderPassEncoder,
        leaf: Leaf,
        mode: Float,
        front: GPUTextureView,
        back: GPUTextureView,
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
        byteBuffer.putFloat(mode)
        byteBuffer.putFloat(0f)
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
     * The leaf's cross-section is a circular arc of radius `len / bend` hinged on the spine, its
     * tangent running `phi` to `phi + bend` - so the sheet swings and curls at once. Heights are
     * width fractions like x, so the arc comes out round rather than stretched.
     */
    override val code = """
struct Uniforms {
    // The leaf's two faces within their cached surfaces: normalised (x1, y1, x2, y2).
    front_rect: vec4<f32>,
    back_rect: vec4<f32>,
    // spine x, direction (+1 resting right of the spine), turn angle, curl
    geom: vec4<f32>,
    // spine-to-edge length, top y, bottom y, surface aspect
    span: vec4<f32>,
    // has_front, has_back, draw mode (0 leaf, 1 shadow), unused
    flags: vec4<f32>,
}

@group(0) @binding(0) var<uniform> flip: Uniforms;
@group(0) @binding(1) var front_tex: texture_2d<f32>;
@group(0) @binding(2) var back_tex: texture_2d<f32>;
@group(0) @binding(3) var flip_sampler: sampler;

const PI: f32 = 3.14159265;

// Eye distance, in the width fractions the geometry is measured in.
const EYE: f32 = 2.6;

// Height of the key light over the spine, in leaf lengths. Under EYE on purpose: a further light
// throws every shadow inside the silhouette casting it.
const LIGHT_HEIGHT: f32 = 2.8;

// How fast a shadow fades as the leaf rises off the page, per leaf length.
const SHADOW_FALLOFF: f32 = 2.6;
const SHADOW_DEPTH: f32 = 0.6;

fn leaf_radius() -> f32 { return flip.span.x / flip.geom.w; }

/// The tangent angle at arc fraction [s] along the leaf - 0 at the spine, 1 at the outer edge.
fn leaf_angle(s: f32) -> f32 { return flip.geom.z + flip.geom.w * s; }

/// Cross-section at tangent angle [b]: distance out from the spine (unsigned) and height.
fn leaf_xz(b: f32) -> vec2<f32> {
    let r = leaf_radius();
    let phi = flip.geom.z;
    return vec2<f32>(r * (sin(b) - sin(phi)), r * (cos(phi) - cos(b)));
}

/// A point on the leaf in centred width fractions, at tangent angle [b] and vertical fraction [v].
fn leaf_point(b: f32, v: f32) -> vec3<f32> {
    let xz = leaf_xz(b);
    let y = mix(flip.span.y, flip.span.z, v);
    return vec3<f32>((flip.geom.x - 0.5) + flip.geom.y * xz.x, (y - 0.5) / flip.span.w, xz.y);
}

/// Centred width fractions back to normalised surface coordinates, under perspective.
fn project(p: vec3<f32>) -> vec2<f32> {
    let s = EYE / (EYE - p.z);
    return vec2<f32>(0.5 + p.x * s, 0.5 + p.y * s * flip.span.w);
}

/// A lamp over the middle of the open book, on the spine itself. Point, not directional, so its
/// shadows run outward: the leaf shades left while it lies left, right while it lies right. Its
/// height scales with the leaf, so the throw holds at any zoom.
fn light_pos() -> vec3<f32> {
    let mid_y = (0.5 * (flip.span.y + flip.span.z) - 0.5) / flip.span.w;
    return vec3<f32>(flip.geom.x - 0.5, mid_y, LIGHT_HEIGHT * flip.span.x);
}

/// Where a point on the leaf lays its shadow down on the page. The clamp only guards a leaf
/// curling as high as its own light, which the turn never approaches.
fn shadow_cast(p: vec3<f32>) -> vec2<f32> {
    let light = light_pos();
    let t = light.z / max(light.z - p.z, 0.25 * light.z);
    return light.xy + t * (p.xy - light.xy);
}

/// How dark the leaf's shadow is where the point casting it sits [z] above the page.
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

/// Colour at arc fraction [s], vertical fraction [v]. cos(b) > 0 faces the viewer; a face with
/// no page behind it is black.
fn leaf_color(s: f32, v: f32) -> vec4<f32> {
    let b = leaf_angle(s);
    let front = cos(b) > 0.0;
    let p = leaf_point(b, v);
    let right = flip.geom.y > 0.0;

    var texel: vec4<f32>;
    if (front) {
        if (flip.flags.x > 0.5) {
            let r = flip.front_rect;
            // The inner edge is the one at the spine, so which edge follows the side it rests on.
            let uv = vec2<f32>(
                mix(select(r.z, r.x, right), select(r.x, r.z, right), s), mix(r.y, r.w, v)
            );
            texel = textureSampleLevel(front_tex, flip_sampler, uv, 0.0);
        } else {
            texel = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    } else {
        if (flip.flags.y > 0.5) {
            let r = flip.back_rect;
            // Mirrored against the front: the back face lies across the spine from it.
            let uv = vec2<f32>(
                mix(select(r.x, r.z, right), select(r.z, r.x, right), s), mix(r.y, r.w, v)
            );
            texel = textureSampleLevel(back_tex, flip_sampler, uv, 0.0);
        } else {
            texel = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }

    // Premultiplied throughout - see premultipliedOutput - so shading scales rgb alone.
    return vec4<f32>(texel.rgb * leaf_shade(p, b, front), texel.a);
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    // Arc fraction along the leaf and vertical fraction down it.
    @location(0) s: f32,
    @location(1) v: f32,
    // Height above the flat page, for the shadow's falloff.
    @location(2) z: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const COLS: u32 = ${COLS}u;
    const ROWS: u32 = ${ROWS}u;
    let quad_index = vertex_index / 6u;
    let vert_in_quad = vertex_index % 6u;
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
    if (flip.flags.z > 0.5) {
        // Shadow: drop the point through the light onto the page. That lands at z = 0, where the
        // perspective divide is the identity.
        let flat = shadow_cast(world);
        screen = vec2<f32>(0.5 + flat.x, 0.5 + flat.y * flip.span.w);
    } else {
        screen = project(world);
    }

    var out: VertexOutput;
    out.position = vec4<f32>(screen.x * 2.0 - 1.0, 1.0 - screen.y * 2.0, 0.0, 1.0);
    out.s = sv.x;
    out.v = sv.y;
    out.z = world.z;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (flip.flags.z > 0.5) {
        // Softened where the leaf lifts away, standing in for a penumbra - but not at the spine,
        // where it still touches the page.
        let fade = smoothstep(0.0, 0.06, 1.0 - in.s) *
            smoothstep(0.0, 0.04, in.v) * smoothstep(0.0, 0.04, 1.0 - in.v);
        // Premultiplied black, so this multiplies what is under it down rather than tinting it.
        return vec4<f32>(0.0, 0.0, 0.0, shadow_alpha(in.z) * fade);
    }
    return leaf_color(in.s, in.v);
}"""
}
