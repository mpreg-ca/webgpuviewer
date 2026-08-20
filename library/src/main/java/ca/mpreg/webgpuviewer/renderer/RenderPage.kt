package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUVertexState
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.RenderPage.TILE_SAMPLER_FS
import ca.mpreg.webgpuviewer.renderer.RenderPage.draw
import ca.mpreg.webgpuviewer.renderer.RenderPage.render
import ca.mpreg.webgpuviewer.renderer.RenderPage.renderFast
import ca.mpreg.webgpuviewer.renderer.RenderPage.renderImage
import ca.mpreg.webgpuviewer.renderer.RenderPage.renderPlain
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Draws a page or a single image into a render pass. Every path that draws a page's live content
 * comes through here: the continuous viewer and the paged viewer when no page turn is in flight.
 *
 * Three shaders, picked per call:
 *  - [render] - box filter minifying, Catmull-Rom magnifying, in linear light. Sharp and
 *    expensive, so bound to a fixed 2x2-tile window - safe only because its one caller,
 *    [TileRenderer]'s tile generation, always targets a single tile-sized destination.
 *  - [renderFast] - one bilinear tap per pixel, also linear-light (so a [TileRenderer] tile
 *    popping in over it never shows a brightness seam). Draws every tile the viewport overlaps
 *    separately, so the viewport can be any size or position without a window falling short.
 *  - [renderPlain] is [renderFast] without the sRGB<->linear round trip, for
 *    [ImagePage.highQuality] false content where that correctness isn't worth the cost.
 *
 * A cached transition's own snapshot is instead composed from [TileRenderer]'s already-generated
 * tiles - see [TileRenderer.blitIfFullyCovered] and [TileRenderer.renderFullyTiled].
 */
object RenderPage {
    private val device get() = WebGpuRenderer.device

    // Thread-local ByteBuffer to avoid per-frame allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    /** One of the three shaders, pipeline built on first use. */
    private class Variant(build: () -> GPURenderPipeline) {
        val pipeline: GPURenderPipeline by lazy(build)
    }

    private val samplerVariant =
        Variant { buildPipeline(TILE_HEADER + TILE_VS_MAIN + TILE_SAMPLER_FS) }
    private val filteredVariant = Variant { buildPipeline(HEADER + VS_MAIN + FILTERED_FS) }
    private val plainVariant = Variant { buildPipeline(TILE_HEADER + TILE_VS_MAIN + TILE_PLAIN_FS) }

    private fun buildPipeline(code: String): GPURenderPipeline {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        return device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = TextureFormat.RGBA8Unorm, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = BlendFactor.SrcAlpha,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    /** Uniforms, texture bindings and the vertex stage's view of the source, shared by both. */
    private const val HEADER = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    tile_size: f32,
    tiles_width: f32,
    tiles_height: f32,
    dst_width: f32,
    dst_height: f32,
}

@group(0) @binding(0) var<uniform> transform: Uniforms;
@group(0) @binding(1) var src_tex0: texture_2d<f32>;
@group(0) @binding(2) var src_tex1: texture_2d<f32>;
@group(0) @binding(3) var src_tex2: texture_2d<f32>;
@group(0) @binding(4) var src_tex3: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn totalDimensions() -> vec2<u32> {
    let w = i32(transform.tiles_width);
    let h = i32(transform.tiles_height);
    if (w <= 0 || h <= 0) {
        return vec2<u32>(0u);
    }

    let dim0 = textureDimensions(src_tex0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(src_tex1).x; }

    var height = dim0.y;
    if (h > 1) { height += textureDimensions(src_tex2).y; }

    return vec2<u32>(width, height);
}

// Shared by both fragment variants: the fast path also filters in linear light now, so both need
// the same sRGB<->linear conversion.
fn to_linear_exact(srgb: vec4<f32>) -> vec4<f32> {
    let c = max(srgb.rgb, vec3<f32>(0.0));
    let lower = c / vec3<f32>(12.92);
    let higher = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let cond = c <= vec3<f32>(0.04045);
    return vec4(select(higher, lower, cond), srgb.a);
}

fn to_srgb_exact(linear_rgb: vec4<f32>) -> vec4<f32> {
    let c = max(linear_rgb.rgb, vec3<f32>(0.0));
    let lower = c * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let cond = c <= vec3<f32>(0.0031308);
    return vec4(select(higher, lower, cond), linear_rgb.a);
}

fn tileLoad(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(src_tex0, pos, 0); }
    if (i == 1) { return textureLoad(src_tex1, pos, 0); }
    if (i == 2) { return textureLoad(src_tex2, pos, 0); }
    return textureLoad(src_tex3, pos, 0);
}

// Fetch by position across the whole quad, picking the tile the position falls in. This is what
// makes filtering work at a tile boundary: the four slots are separate textures, so anything
// that resolves within one tile has no access to its neighbour's edge texels.
fn totalLoad(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(transform.tile_size);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;

    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad(idx, pos0);
}
"""

    private const val VS_MAIN = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );

    let uv = uvs[vertex_index];

    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(totalDimensions());

    // Calculate destination canvas pixel position
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);

    // Convert pixel coordinate to WebGPU NDC Space:
    // X goes from [-1.0, 1.0] (left to right)
    // Y goes from [1.0, -1.0] (top to bottom)
    let ndc_x = (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / dst_size_f.y) * 2.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uv;
    return out;
}
"""

    /**
     * Uniforms, single-texture binding and vertex stage shared by [renderFast]/[renderPlain]'s
     * per-tile draws - no [tile_size]/[tiles_width]/[tiles_height] bookkeeping, since a draw
     * through here is always exactly one tile.
     */
    private const val TILE_HEADER = """
struct TileUniforms {
    offset: vec2<f32>,
    scale: f32,
    dst_width: f32,
    dst_height: f32,
}

@group(0) @binding(0) var<uniform> transform: TileUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;

struct TileVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

fn tile_to_linear_exact(srgb: vec4<f32>) -> vec4<f32> {
    let c = max(srgb.rgb, vec3<f32>(0.0));
    let lower = c / vec3<f32>(12.92);
    let higher = pow((c + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
    let cond = c <= vec3<f32>(0.04045);
    return vec4(select(higher, lower, cond), srgb.a);
}

fn tile_to_srgb_exact(linear_rgb: vec4<f32>) -> vec4<f32> {
    let c = max(linear_rgb.rgb, vec3<f32>(0.0));
    let lower = c * vec3<f32>(12.92);
    let higher = vec3<f32>(1.055) * pow(c, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    let cond = c <= vec3<f32>(0.0031308);
    return vec4(select(higher, lower, cond), linear_rgb.a);
}
"""

    private const val TILE_VS_MAIN = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> TileVertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let uv = uvs[vertex_index];
    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(textureDimensions(src_tex));
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);

    var out: TileVertexOutput;
    out.position = vec4<f32>(
        (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0,
        1.0 - (pixel_pos.y / dst_size_f.y) * 2.0,
        0.0, 1.0
    );
    out.uv = uv;
    return out;
}
"""

    /** Fragment stage for [renderFast]: one bilinear resolve per pixel, in linear light. */
    private const val TILE_SAMPLER_FS = """
@fragment
fn fs_main(in: TileVertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(src_tex));
    let pos = in.uv * size;
    let p = pos - 0.5;
    let base = floor(p);

    let max_coord = vec2<i32>(size) - 1;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), max_coord);
    let i1 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), max_coord);
    let f = p - base;

    let c00 = tile_to_linear_exact(textureLoad(src_tex, vec2<i32>(i0.x, i0.y), 0));
    let c10 = tile_to_linear_exact(textureLoad(src_tex, vec2<i32>(i1.x, i0.y), 0));
    let c01 = tile_to_linear_exact(textureLoad(src_tex, vec2<i32>(i0.x, i1.y), 0));
    let c11 = tile_to_linear_exact(textureLoad(src_tex, vec2<i32>(i1.x, i1.y), 0));

    let linear_col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    let col = tile_to_srgb_exact(linear_col);
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /**
     * Fragment stage for [renderPlain]: [TILE_SAMPLER_FS]'s bilinear tap with the sRGB<->linear
     * round trip removed - see the class doc for the tradeoff this makes.
     */
    private const val TILE_PLAIN_FS = """
@fragment
fn fs_main(in: TileVertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(src_tex));
    let pos = in.uv * size;
    let p = pos - 0.5;
    let base = floor(p);

    let max_coord = vec2<i32>(size) - 1;
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), max_coord);
    let i1 = clamp(vec2<i32>(base) + 1, vec2<i32>(0), max_coord);
    let f = p - base;

    let c00 = textureLoad(src_tex, vec2<i32>(i0.x, i0.y), 0);
    let c10 = textureLoad(src_tex, vec2<i32>(i1.x, i0.y), 0);
    let c01 = textureLoad(src_tex, vec2<i32>(i0.x, i1.y), 0);
    let c11 = textureLoad(src_tex, vec2<i32>(i1.x, i1.y), 0);

    let col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    return vec4<f32>(col.rgb * col.a, col.a);
}
"""

    /** Fragment stage for [render]: box filter when minifying, Catmull-Rom when magnifying. */
    private const val FILTERED_FS = """
fn catmull_rom_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;

    return array<f32, 4>(
        -0.5 * t3 + t2 - 0.5 * t,          // Weight 0 (Negative lobe)
         1.5 * t3 - 2.5 * t2 + 1.0,        // Weight 1 (Primary influence)
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,    // Weight 2 (Primary influence)
         0.5 * t3 - 0.5 * t2               // Weight 3 (Negative lobe)
    );
}

fn catmull_rom_fast_unrolled(
    tex: texture_2d<f32>,
    p_start: vec2<i32>,
    wx: array<f32, 4>,
    wy: array<f32, 4>
) -> vec4<f32> {
    let r0 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y), 0)) * wx[3];
    let r1 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 1), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 1), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 1), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 1), 0)) * wx[3];
    let r2 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 2), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 2), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 2), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 2), 0)) * wx[3];
    let r3 = to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x,     p_start.y + 3), 0)) * wx[0]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 1, p_start.y + 3), 0)) * wx[1]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 2, p_start.y + 3), 0)) * wx[2]
           + to_linear_exact(textureLoad(tex, vec2<i32>(p_start.x + 3, p_start.y + 3), 0)) * wx[3];

    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn load_safe_linear(pos: vec2<i32>, max_coord: vec2<i32>) -> vec4<f32> {
    let clamped = clamp(pos, vec2<i32>(0), max_coord);
    return to_linear_exact(totalLoad(clamped));
}

fn catmull_rom_slow_unrolled(
    start_i: vec2<i32>,
    max_coord: vec2<i32>,
    wx: array<f32, 4>,
    wy: array<f32, 4>
) -> vec4<f32> {
    let r0 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y), max_coord) * wx[3];
    let r1 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 1), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 1), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 1), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 1), max_coord) * wx[3];
    let r2 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 2), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 2), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 2), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 2), max_coord) * wx[3];
    let r3 = load_safe_linear(vec2<i32>(start_i.x,     start_i.y + 3), max_coord) * wx[0]
           + load_safe_linear(vec2<i32>(start_i.x + 1, start_i.y + 3), max_coord) * wx[1]
           + load_safe_linear(vec2<i32>(start_i.x + 2, start_i.y + 3), max_coord) * wx[2]
           + load_safe_linear(vec2<i32>(start_i.x + 3, start_i.y + 3), max_coord) * wx[3];
    return r0 * wy[0] + r1 * wy[1] + r2 * wy[2] + r3 * wy[3];
}

fn textureSampleCatmullRom(uv: vec2<f32>) -> vec4<f32> {
    let tex_size_u = totalDimensions();
    let tex_size = vec2<f32>(tex_size_u);
    let pixel_coord = uv * tex_size - 0.5;
    let base_coord = vec2<i32>(floor(pixel_coord));
    let f = fract(pixel_coord);

    let wx = catmull_rom_weights(f.x);
    let wy = catmull_rom_weights(f.y);
    let max_coord = vec2<i32>(tex_size_u) - 1;

    let ts = i32(transform.tile_size);

    let start_i = base_coord - vec2<i32>(1); // Top-left
    let end_i   = base_coord + vec2<i32>(2); // Bottom-right

    let canvas_in_bounds = start_i.x >= 0 && start_i.y >= 0 && end_i.x <= max_coord.x && end_i.y <= max_coord.y;
    let tile_TL = start_i / ts;
    let tile_BR = end_i / ts;
    let is_single_tile = all(tile_TL == tile_BR) && canvas_in_bounds;

    var final_color_linear = vec4<f32>(0.0);

    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        let p_start = start_i + local_offset;

        if (idx == 0) {
            final_color_linear = catmull_rom_fast_unrolled(src_tex0, p_start, wx, wy);
        } else if (idx == 1) {
            final_color_linear = catmull_rom_fast_unrolled(src_tex1, p_start, wx, wy);
        } else if (idx == 2) {
            final_color_linear = catmull_rom_fast_unrolled(src_tex2, p_start, wx, wy);
        } else {
            final_color_linear = catmull_rom_fast_unrolled(src_tex3, p_start, wx, wy);
        }
    } else {
        final_color_linear = catmull_rom_slow_unrolled(start_i, max_coord, wx, wy);
    }

    return clamp(to_srgb_exact(final_color_linear), vec4(0.0), vec4(1.0));
}

fn loop_over_tile(
    tex: texture_2d<f32>,
    start_i: vec2<i32>,
    end_i: vec2<i32>,
    src_start: vec2<f32>,
    src_end: vec2<f32>,
    local_offset: vec2<i32>
) -> vec4<f32> {
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    for (var y: i32 = start_i.y; y < end_i.y; y++) {
        let y_f = f32(y);

        var y_overlap = 1.0;
        if (y == start_i.y) {
            y_overlap = min(y_f + 1.0, src_end.y) - src_start.y;
        } else if (y == end_i.y - 1) {
            y_overlap = src_end.y - max(y_f, src_start.y);
        }
        y_overlap = max(0.0, y_overlap);

        let py = y + local_offset.y;

        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);

            var x_overlap = 1.0;
            if (x == start_i.x) {
                x_overlap = min(x_f + 1.0, src_end.x) - src_start.x;
            } else if (x == end_i.x - 1) {
                x_overlap = src_end.x - max(x_f, src_start.x);
            }
            x_overlap = max(0.0, x_overlap);

            let weight = x_overlap * y_overlap;
            let px = x + local_offset.x;

            let texel = to_linear_exact(textureLoad(tex, vec2<i32>(px, py), 0));
            color_sum += texel * weight;
            weight_sum += weight;
        }
    }
    return color_sum / max(weight_sum, 0.0001);
}

fn downsample(src_start: vec2<f32>, scale: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions());
    let src_end = src_start + scale;

    let start_i = vec2<i32>(clamp(floor(src_start), vec2<f32>(0.0), src_size_f));
    let end_i   = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));

    let ts = i32(transform.tile_size);

    let tile_TL = start_i / ts;
    let tile_BR = (end_i - 1) / ts;

    let in_bounds = start_i.x >= 0 && start_i.y >= 0 && (end_i.x - 1) < ts * 2 && (end_i.y - 1) < ts * 2;
    let is_single_tile = all(tile_TL == tile_BR) && in_bounds;

    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;

        var avg_color = vec4<f32>(0.0);

        if (idx == 0) {
            avg_color = loop_over_tile(src_tex0, start_i, end_i, src_start, src_end, local_offset);
        } else if (idx == 1) {
            avg_color = loop_over_tile(src_tex1, start_i, end_i, src_start, src_end, local_offset);
        } else if (idx == 2) {
            avg_color = loop_over_tile(src_tex2, start_i, end_i, src_start, src_end, local_offset);
        } else {
            avg_color = loop_over_tile(src_tex3, start_i, end_i, src_start, src_end, local_offset);
        }

        return to_srgb_exact(avg_color);
    } else {
        for (var y: i32 = start_i.y; y < end_i.y; y++) {
            let y_f = f32(y);
            var y_overlap = 1.0;
            if (y == start_i.y) {
                y_overlap = min(y_f + 1.0, src_end.y) - src_start.y;
            } else if (y == end_i.y - 1) {
                y_overlap = src_end.y - max(y_f, src_start.y);
            }
            y_overlap = max(0.0, y_overlap);

            for (var x: i32 = start_i.x; x < end_i.x; x++) {
                let x_f = f32(x);
                var x_overlap = 1.0;
                if (x == start_i.x) {
                    x_overlap = min(x_f + 1.0, src_end.x) - src_start.x;
                } else if (x == end_i.x - 1) {
                    x_overlap = src_end.x - max(x_f, src_start.x);
                }
                x_overlap = max(0.0, x_overlap);

                let weight = x_overlap * y_overlap;
                let texel = to_linear_exact(totalLoad(vec2<i32>(x, y)));
                color_sum += texel * weight;
                weight_sum += weight;
            }
        }

        return to_srgb_exact(color_sum / max(weight_sum, 0.0001));
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions());
    let scale_factor = 1.0 / transform.scale;
    let scale_vec = vec2<f32>(scale_factor);

    var col = vec4<f32>(0.0);

    if (scale_factor > 1.0) {
        // downsample expects src_start (position in the source image in pixels)
        let src_start = in.uv * src_size_f;
        col = downsample(src_start, scale_vec);
    } else {
        col = textureSampleCatmullRom(in.uv);
    }

    return vec4<f32>(col.rgb * col.a, col.a);
}"""

    /**
     * Draw an image into [pass] with the filtered shader. Takes a pass rather than an encoder so
     * a whole frame's draws can share one - a pass per image costs an attachment load/store each,
     * which dominates frame cost on tile-based GPUs. Opening/ending the pass is the caller's job.
     */
    internal fun render(
        pass: GPURenderPassEncoder, image: Image, dst: GPUTexture, x: Float, y: Float, scale: Float
    ) = renderImage(pass, image, dst, x, y, scale, filteredVariant)

    /** Draw an image into [pass] with the sampler shader. */
    internal fun renderFast(
        pass: GPURenderPassEncoder, image: Image, dst: GPUTexture, x: Float, y: Float, scale: Float
    ) = renderImageTiled(pass, image, dst, x, y, scale, samplerVariant)

    /** Draw an image into [pass] with the plain (non-linear-light) sampler shader. */
    internal fun renderPlain(
        pass: GPURenderPassEncoder, image: Image, dst: GPUTexture, x: Float, y: Float, scale: Float
    ) = renderImageTiled(pass, image, dst, x, y, scale, plainVariant)

    /** Draw a page into [pass] with the sampler shader. */
    internal fun renderFast(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) = renderPage(pass, page, dst, x, y, scale, samplerVariant)

    /** Draw a page into [pass] with the plain (non-linear-light) sampler shader. */
    internal fun renderPlain(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) = renderPage(pass, page, dst, x, y, scale, plainVariant)

    /**
     * Draw just a page's per-image background colour, skipping the image itself - for a caller
     * that knows [TileRenderer] already covers it and can skip [renderFast]. The background's
     * alpha depends on live pan/scale, so it's drawn every frame regardless of tile coverage.
     */
    internal fun renderBackground(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) = renderPage(pass, page, dst, x, y, scale, null)

    private fun renderImage(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        variant: Variant
    ) {
        val res = image.prepareForRender(dst, x, y, scale) ?: return
        draw(pass, image, dst, res, variant)
    }

    /** As [renderImage], for [renderFast]/[renderPlain] - draws every tile separately. */
    private fun renderImageTiled(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        variant: Variant
    ) {
        for (tile in image.prepareTilesForRender(dst, x, y, scale)) {
            drawTile(pass, dst, tile, variant)
        }
    }

    private fun renderPage(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        variant: Variant?
    ) {
        // A snapshot is captured on the main thread and drawn later, so the page may have been
        // evicted in between. Its images' buffers are already destroyed, and touching one throws.
        if (page.destroyed || page.images.all { it == null }) return

        // Use the current animation frame for single-image pages; fall back to the full
        // images list for dual-page spreads (which are never animated).
        val renderImages = if (page.images.size == 1) listOf(page.image) else page.images

        // For each image, prepare render and draw background + image
        renderImages.forEach { image ->
            image ?: return@forEach
            val offsetX = when (image.position) {
                Image.Position.LEFT -> (-0.5f * image.width) / dst.width
                Image.Position.RIGHT -> (0.5f * image.width) / dst.width
                Image.Position.SINGLE -> 0f
            }
            if (image.mipmaps.isEmpty()) return@forEach
            val placeX = page.x + x + offsetX
            val placeY = page.y + y
            val placeScale = page.scale * scale
            val rect = image.placement(dst, placeX, placeY, placeScale)

            // Draw background color behind the image
            val parent = page.parent
            val minScale = page.minScale
            val homeScale = page.homeScale
            val currentScale = page.scale * scale

            val fadeDistancePixels = 200f
            val imageSize = (page.width.coerceAtLeast(page.height)).toFloat()

            fun proximity(anchorScale: Float): Float {
                if (anchorScale <= 0f) return 0f
                val deltaPixels = abs(imageSize * (currentScale - anchorScale))
                return (1f - deltaPixels / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundProximity(value: Float, lo: Float, hi: Float, pixelsPerUnit: Float): Float {
                val overflow = when {
                    value < lo -> lo - value
                    value > hi -> value - hi
                    else -> return 1f
                }
                return (1f - overflow * pixelsPerUnit / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundsProximityAt(anchorScale: Float): Float {
                if (parent == null || anchorScale <= 0f) return 0f
                val maxX = page.maxX(anchorScale)
                val minY = page.minY(anchorScale)
                val maxY = page.maxY(anchorScale)
                val pixelsPerUnitX = parent.width.toFloat() * anchorScale
                val pixelsPerUnitY = parent.height.toFloat() * anchorScale
                return min(
                    boundProximity(page.x, -maxX, maxX, pixelsPerUnitX),
                    boundProximity(page.y, minY, maxY, pixelsPerUnitY)
                )
            }

            val bgAlpha = if (parent != null) {
                val homeProximity = min(proximity(homeScale), boundsProximityAt(homeScale))
                val minProximity = min(proximity(minScale), boundsProximityAt(minScale))
                max(homeProximity, minProximity)
            } else {
                1f
            }

            val origA = (image.backgroundColor ushr 24) and 0xFF
            val a = (origA * bgAlpha).toInt()

            if (a > 0) {
                val x1 = if (image.position == Image.Position.SINGLE) 0f else rect[0]
                val x2 = if (image.position == Image.Position.SINGLE) 1f else rect[2]
                val origR = (image.backgroundColor shr 16) and 0xFF
                val origG = (image.backgroundColor shr 8) and 0xFF
                val origB = image.backgroundColor and 0xFF
                val r = (origR * bgAlpha).toInt()
                val g = (origG * bgAlpha).toInt()
                val b = (origB * bgAlpha).toInt()
                val bgColor = (a shl 24) or (r shl 16) or (g shl 8) or b
                Draw.rect(pass, x1, 0f, x2, 1f, bgColor)
            }

            // Render the image on top, one tile at a time - skipped when variant is null
            // (background-only draw, used when TileRenderer already covers the image itself).
            if (variant != null) {
                for (tile in image.prepareTilesForRender(dst, placeX, placeY, placeScale)) {
                    drawTile(pass, dst, tile, variant)
                }
            }
        }
    }

    private fun draw(
        pass: GPURenderPassEncoder,
        image: Image,
        dst: GPUTexture,
        res: Image.MipMapForDraw,
        variant: Variant
    ) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(res.x)
        byteBuffer.putFloat(res.y)
        byteBuffer.putFloat(res.scale)
        byteBuffer.putFloat(res.mipmap.tilesize.toFloat())
        byteBuffer.putFloat(res.mipmap.tilesCols.toFloat())
        byteBuffer.putFloat(res.mipmap.tilesRows.toFloat())
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.flip()

        device.queue.writeBuffer(image.buffer, 0, byteBuffer)

        val pipeline = variant.pipeline
        val entries = arrayOf(
            GPUBindGroupEntry(0, buffer = image.buffer),
        ).plus(res.quad.tileViews.mapIndexed { i, view ->
            GPUBindGroupEntry(1 + i, textureView = view)
        })

        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(layout = pipeline.getBindGroupLayout(0), entries = entries)
            )
        )

        pass.draw(6)
    }

    /**
     * Draw one tile from [Image.prepareTilesForRender], into its own persistent uniform buffer
     * rather than the shared scratch one [draw] uses - see [Mipmap.TileRect]'s doc for why.
     */
    private fun drawTile(
        pass: GPURenderPassEncoder, dst: GPUTexture, tile: Image.TileForDraw, variant: Variant
    ) {
        val byteBuffer = byteBufferLocal.get()
        byteBuffer.clear()
        byteBuffer.putFloat(tile.x)
        byteBuffer.putFloat(tile.y)
        byteBuffer.putFloat(tile.scale)
        byteBuffer.putFloat(dst.width.toFloat())
        byteBuffer.putFloat(dst.height.toFloat())
        byteBuffer.flip()

        device.queue.writeBuffer(tile.uniform, 0, byteBuffer)

        val pipeline = variant.pipeline
        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = tile.uniform),
                        GPUBindGroupEntry(1, textureView = tile.view),
                    )
                )
            )
        )
        pass.draw(6)
    }
}
