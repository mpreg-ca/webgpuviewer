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
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object TransitionFade : Transition() {
    override val code = """
struct Uniforms {
    offset1: vec2<f32>,
    scale1: f32,
    tile_size1: f32,
    tiles_width1: f32,
    tiles_height1: f32,
    offset2: vec2<f32>,
    scale2: f32,
    tile_size2: f32,
    tiles_width2: f32,
    tiles_height2: f32,
    dst_width: f32,
    dst_height: f32,
    blend: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var tex1_0: texture_2d<f32>;
@group(0) @binding(2) var tex1_1: texture_2d<f32>;
@group(0) @binding(3) var tex1_2: texture_2d<f32>;
@group(0) @binding(4) var tex1_3: texture_2d<f32>;
@group(0) @binding(5) var tex2_0: texture_2d<f32>;
@group(0) @binding(6) var tex2_1: texture_2d<f32>;
@group(0) @binding(7) var tex2_2: texture_2d<f32>;
@group(0) @binding(8) var tex2_3: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv1: vec2<f32>,
    @location(1) uv2: vec2<f32>,
};

fn tileLoad1(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(tex1_0, pos, 0); }
    if (i == 1) { return textureLoad(tex1_1, pos, 0); }
    if (i == 2) { return textureLoad(tex1_2, pos, 0); }
    return textureLoad(tex1_3, pos, 0);
}

fn tileLoad2(i: i32, pos: vec2<i32>) -> vec4<f32> {
    if (i == 0) { return textureLoad(tex2_0, pos, 0); }
    if (i == 1) { return textureLoad(tex2_1, pos, 0); }
    if (i == 2) { return textureLoad(tex2_2, pos, 0); }
    return textureLoad(tex2_3, pos, 0);
}

fn totalDimensions1() -> vec2<u32> {
    let w = i32(u.tiles_width1);
    let h = i32(u.tiles_height1);
    if (w <= 0 || h <= 0) { return vec2<u32>(0u); }
    let dim0 = textureDimensions(tex1_0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(tex1_1).x; }
    var height = dim0.y;
    if (h > 1) { height += textureDimensions(tex1_2).y; }
    return vec2<u32>(width, height);
}

fn totalDimensions2() -> vec2<u32> {
    let w = i32(u.tiles_width2);
    let h = i32(u.tiles_height2);
    if (w <= 0 || h <= 0) { return vec2<u32>(0u); }
    let dim0 = textureDimensions(tex2_0);
    var width = dim0.x;
    if (w > 1) { width += textureDimensions(tex2_1).x; }
    var height = dim0.y;
    if (h > 1) { height += textureDimensions(tex2_2).y; }
    return vec2<u32>(width, height);
}

fn totalLoad1(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(u.tile_size1);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;
    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad1(idx, pos0);
}

fn totalLoad2(pos: vec2<i32>) -> vec4<f32> {
    let ts = i32(u.tile_size2);
    let tile_x = select(0, 1, pos.x >= ts);
    let tile_y = select(0, 1, pos.y >= ts);
    let idx = tile_y * 2 + tile_x;
    let pos0 = pos - vec2<i32>(tile_x, tile_y) * ts;
    return tileLoad2(idx, pos0);
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

fn sampleImage1(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(totalDimensions1());
    let pos = vec2<i32>(uv * vec2<f32>(size));
    if (pos.x < 0 || pos.y < 0 || pos.x >= size.x || pos.y >= size.y) {
        return vec4<f32>(0.0);
    }
    return totalLoad1(pos);
}

fn sampleImage2(uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(totalDimensions2());
    let pos = vec2<i32>(uv * vec2<f32>(size));
    if (pos.x < 0 || pos.y < 0 || pos.x >= size.x || pos.y >= size.y) {
        return vec4<f32>(0.0);
    }
    return totalLoad2(pos);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );

    let uv = uvs[vertex_index];
    let dst_size_f = vec2<f32>(u.dst_width, u.dst_height);
    
    // Use full screen quad
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    
    // Calculate UV for each image based on their transforms
    let src_size1 = vec2<f32>(totalDimensions1());
    let src_size2 = vec2<f32>(totalDimensions2());
    
    // Convert screen UV to image UV
    let screen_pos = uv * dst_size_f;
    let uv1 = (screen_pos / u.scale1 - u.offset1 * dst_size_f) / src_size1;
    let uv2 = (screen_pos / u.scale2 - u.offset2 * dst_size_f) / src_size2;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv1 = uv1;
    out.uv2 = uv2;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col1 = sampleImage1(in.uv1);
    let col2 = sampleImage2(in.uv2);
    
    // Convert to linear for proper blending
    let linear1 = to_linear_exact(col1);
    let linear2 = to_linear_exact(col2);
    
    // Blend in linear space
    let t = u.blend;
    let blended = mix(linear1, linear2, t);
    
    // Convert back to sRGB
    let result = to_srgb_exact(blended);
    return vec4<f32>(result.rgb * result.a, result.a);
}"""

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        val image1 = page1.image ?: return
        val image2 = page2.image ?: return
        val res1 = image1.prepareForRender(dst, page1.x, page1.y, page1.scale) ?: return
        val res2 = image2.prepareForRender(dst, page2.x, page2.y, page2.scale) ?: return

        // blend: 0 = fully page1, 1 = fully page2
        val blend = if (frac > 0f) frac else -frac

        val byteBuffer = ByteBuffer.allocateDirect(60).apply {
            order(ByteOrder.nativeOrder())
            // Image 1
            putFloat(0, res1.x)
            putFloat(4, res1.y)
            putFloat(8, res1.scale)
            putFloat(12, res1.mipmap.tilesize.toFloat())
            putFloat(16, res1.mipmap.tilesCols.toFloat())
            putFloat(20, res1.mipmap.tilesRows.toFloat())
            // Image 2
            putFloat(24, res2.x)
            putFloat(28, res2.y)
            putFloat(32, res2.scale)
            putFloat(36, res2.mipmap.tilesize.toFloat())
            putFloat(40, res2.mipmap.tilesCols.toFloat())
            putFloat(44, res2.mipmap.tilesRows.toFloat())
            // Shared
            putFloat(48, dst.width.toFloat())
            putFloat(52, dst.height.toFloat())
            putFloat(56, blend)
        }

        device.queue.writeBuffer(image1.buffer, 0, byteBuffer)

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
                        GPUBindGroupEntry(0, buffer = image1.buffer),
                    ).plus(res1.quad.tileViews.mapIndexed { i, view ->
                        GPUBindGroupEntry(1 + i, textureView = view)
                    }).plus(res2.quad.tileViews.mapIndexed { i, view ->
                        GPUBindGroupEntry(5 + i, textureView = view)
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }
}
