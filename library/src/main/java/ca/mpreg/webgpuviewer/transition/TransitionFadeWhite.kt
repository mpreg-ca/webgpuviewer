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

object TransitionFadeWhite : Transition() {
    override val code = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    tile_size: f32,
    tiles_width: f32,
    tiles_height: f32,
    dst_width: f32,
    dst_height: f32,
    fade_to_white: f32,
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
    if (w <= 0 || h <= 0) { return vec2<u32>(0u); }
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

fn sampleImage(uv: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions());
    let pos = vec2<i32>(uv * src_size_f);
    let size = vec2<i32>(totalDimensions());
    if (pos.x < 0 || pos.y < 0 || pos.x >= size.x || pos.y >= size.y) {
        return vec4<f32>(0.0);
    }
    return totalLoad(pos);
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
    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(totalDimensions());
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);
    let ndc_x = (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_pos.y / dst_size_f.y) * 2.0;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let col = sampleImage(in.uv);
    let t = transform.fade_to_white;
    // Mix image color with white
    let white = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let result = mix(col, white, t);
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
        // frac goes from 0 to 1 (forward) or 0 to -1 (backward)
        val t = if (frac > 0f) frac else -frac
        
        // First half: fade current to white
        // Second half: fade white to next
        if (t < 0.5f) {
            // Fade page1 to white (t goes 0 -> 0.5, fadeToWhite goes 0 -> 1)
            val fadeToWhite = t * 2f
            if (frac > 0f) {
                renderPage(page1, encoder, dst, fadeToWhite)
            } else {
                renderPage(page1, encoder, dst, fadeToWhite)
            }
        } else {
            // Fade white to page2 (t goes 0.5 -> 1, fadeToWhite goes 1 -> 0)
            val fadeToWhite = (1f - t) * 2f
            if (frac > 0f) {
                renderPage(page2, encoder, dst, fadeToWhite)
            } else {
                renderPage(page2, encoder, dst, fadeToWhite)
            }
        }
    }

    private fun renderPage(
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        fadeToWhite: Float,
    ) {
        val image = page.image ?: return
        val res = image.prepareForRender(dst, page.x, page.y, page.scale) ?: return

        val byteBuffer = ByteBuffer.allocateDirect(36).apply {
            order(ByteOrder.nativeOrder())
            putFloat(0, res.x)
            putFloat(4, res.y)
            putFloat(8, res.scale)
            putFloat(12, res.mipmap.tilesize.toFloat())
            putFloat(16, res.mipmap.tilesCols.toFloat())
            putFloat(20, res.mipmap.tilesRows.toFloat())
            putFloat(24, dst.width.toFloat())
            putFloat(28, dst.height.toFloat())
            putFloat(32, fadeToWhite)
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
                    ).plus(res.quad.tileViews.mapIndexed { i, view ->
                        GPUBindGroupEntry(1 + i, textureView = view)
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }
}
