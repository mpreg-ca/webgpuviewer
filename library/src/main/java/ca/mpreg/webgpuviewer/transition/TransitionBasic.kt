package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TransitionBasic : Transition() {
    override val code = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    tile_size: f32,
    tiles_width: f32,
    tiles_height: f32,
    dst_width: f32,
    dst_height: f32,
    page_flip: f32,
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

fn tileLoad(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(src_tex0, pos, 0); }
    if (i == 1) { return textureLoad(src_tex1, pos, 0); }
    if (i == 2) { return textureLoad(src_tex2, pos, 0); }
    return textureLoad(src_tex3, pos, 0);
}

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

fn totalLoad(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(transform.tile_size);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;

    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad(idx, pos0);
}

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
    if (pos.x >= 0 && pos.x <= max_coord.x && pos.y >= 0 && pos.y <= max_coord.y) {
        return to_linear_exact(totalLoad(pos));
    }
    return vec4<f32>(0.0);
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

    internal fun render(
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) {
        val image = page.image ?: return
        render(image, encoder, dst, page.x + x, page.y + y, page.scale * scale)
    }

    internal fun render(
        image: Image, encoder: GPUCommandEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float
    ) {
        val res = image.prepareForRender(dst, x, y, scale) ?: return

        val byteBuffer = ByteBuffer.allocateDirect(32).apply {
            order(ByteOrder.nativeOrder())
            putFloat(0, res.x)
            putFloat(4, res.y)
            putFloat(8, res.scale)
            putFloat(12, res.mipmap.tilesize.toFloat())
            putFloat(16, res.mipmap.tilesCols.toFloat())
            putFloat(20, res.mipmap.tilesRows.toFloat())
            putFloat(24, dst.width.toFloat())
            putFloat(28, dst.height.toFloat())
        }

        device.queue.writeBuffer(image.buffer, 0, byteBuffer)

        val pass = encoder.beginRenderPass(
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

        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = image.buffer),
                    ).plus(res.quad.tiles.mapIndexed { i, value ->
                        GPUBindGroupEntry(1 + i, textureView = value.createView())
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        if (frac > 0f) {
            render(page2, encoder, dst, (1f - frac) / page2.scale, 0f, 1f)
            render(page1, encoder, dst, -frac / page1.scale, 0f, 1f)
        } else {
            render(page2, encoder, dst, -(frac + 1f) / page2.scale, 0f, 1f)
            render(page1, encoder, dst, -frac / page1.scale, 0f, 1f)
        }
    }

    object Vertical : Transition() {
        override val pipeline = TransitionBasic.pipeline

        override fun render(
            page1: ImagePage,
            page2: ImagePage,
            encoder: GPUCommandEncoder,
            dst: GPUTexture,
            frac: Float,
            pos1: Offset,
            pos2: Offset,
        ) {
            if (frac > 0f) {
                render(page1, encoder, dst, 0f, -frac / page1.scale, 1f)
                render(page2, encoder, dst, 0f, (1f - frac) / page2.scale, 1f)
            } else {
                render(page2, encoder, dst, 0f, -(frac + 1f) / page2.scale, 1f)
                render(page1, encoder, dst, 0f, -frac / page1.scale, 1f)
            }
        }
    }
}
