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
import kotlin.math.cos
import kotlin.math.sin

object TransitionCube : Transition() {
    override val code = """
struct Uniforms {
    offset: vec2<f32>,
    scale: f32,
    tile_size: f32,
    tiles_width: f32,
    tiles_height: f32,
    dst_width: f32,
    dst_height: f32,
    transform_mat: mat4x4<f32>,
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
        if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
        else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
        y_overlap = max(0.0, y_overlap);
        let py = y + local_offset.y;
        for (var x: i32 = start_i.x; x < end_i.x; x++) {
            let x_f = f32(x);
            var x_overlap = 1.0;
            if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
            else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
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
    let end_i = vec2<i32>(clamp(ceil(src_end), vec2<f32>(0.0), src_size_f));
    let ts = i32(transform.tile_size);
    let tile_TL = start_i / ts;
    let tile_BR = (end_i - 1) / ts;
    let in_bounds = start_i.x >= 0 && start_i.y >= 0 && (end_i.x - 1) < ts * 2 && (end_i.y - 1) < ts * 2;
    let is_single_tile = all(tile_TL == tile_BR) && in_bounds;

    if (is_single_tile) {
        let idx = tile_TL.y * 2 + tile_TL.x;
        let local_offset = -tile_TL * ts;
        var avg_color = vec4<f32>(0.0);
        if (idx == 0) { avg_color = loop_over_tile(src_tex0, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 1) { avg_color = loop_over_tile(src_tex1, start_i, end_i, src_start, src_end, local_offset); }
        else if (idx == 2) { avg_color = loop_over_tile(src_tex2, start_i, end_i, src_start, src_end, local_offset); }
        else { avg_color = loop_over_tile(src_tex3, start_i, end_i, src_start, src_end, local_offset); }
        return to_srgb_exact(avg_color);
    } else {
        var color_sum = vec4<f32>(0.0);
        var weight_sum = 0.0;
        for (var y: i32 = start_i.y; y < end_i.y; y++) {
            let y_f = f32(y);
            var y_overlap = 1.0;
            if (y == start_i.y) { y_overlap = min(y_f + 1.0, src_end.y) - src_start.y; }
            else if (y == end_i.y - 1) { y_overlap = src_end.y - max(y_f, src_start.y); }
            y_overlap = max(0.0, y_overlap);
            for (var x: i32 = start_i.x; x < end_i.x; x++) {
                let x_f = f32(x);
                var x_overlap = 1.0;
                if (x == start_i.x) { x_overlap = min(x_f + 1.0, src_end.x) - src_start.x; }
                else if (x == end_i.x - 1) { x_overlap = src_end.x - max(x_f, src_start.x); }
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

fn sampleImage(uv: vec2<f32>) -> vec4<f32> {
    let src_size_f = vec2<f32>(totalDimensions());
    let scale_factor = 1.0 / transform.scale;
    if (scale_factor > 1.0) {
        let src_start = uv * src_size_f;
        return downsample(src_start, vec2<f32>(scale_factor));
    }
    let pos = vec2<i32>(uv * src_size_f);
    let size = vec2<i32>(totalDimensions());
    if (pos.x < 0 || pos.y < 0 || pos.x >= size.x || pos.y >= size.y) {
        return vec4<f32>(0.0);
    }
    return totalLoad(pos);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    const COLS: u32 = 32u;
    const ROWS: u32 = 32u;
    let quad_index = vertex_index / 6u;
    let vert_in_quad = vertex_index % 6u;
    let col = quad_index % COLS;
    let row = quad_index / COLS;

    let x0 = f32(col) / f32(COLS);
    let x1 = f32(col + 1u) / f32(COLS);
    let y0 = f32(row) / f32(ROWS);
    let y1 = f32(row + 1u) / f32(ROWS);

    var uv: vec2<f32>;
    switch (vert_in_quad) {
        case 0u: { uv = vec2<f32>(x0, y0); }
        case 1u: { uv = vec2<f32>(x0, y1); }
        case 2u: { uv = vec2<f32>(x1, y0); }
        case 3u: { uv = vec2<f32>(x1, y0); }
        case 4u: { uv = vec2<f32>(x0, y1); }
        default: { uv = vec2<f32>(x1, y1); }
    }

    let dst_size_f = vec2<f32>(transform.dst_width, transform.dst_height);
    let src_size_f = vec2<f32>(totalDimensions());

    // Phase packed in mat[3][2]: 0=flat, 1=fully transformed
    let phase = transform.transform_mat[3][2];

    // Flat position: image at its current transform
    let pixel_pos = transform.scale * (transform.offset * dst_size_f + uv * src_size_f);
    let flat_ndc = vec2<f32>(
        (pixel_pos.x / dst_size_f.x) * 2.0 - 1.0,
        1.0 - (pixel_pos.y / dst_size_f.y) * 2.0
    );

    // Transformed position: matrix maps unit quad to clip space, perspective via W
    let local_pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    let transformed = transform.transform_mat * local_pos;
    let trans_ndc = transformed.xy / transformed.w;

    // Interpolate between flat and transformed
    let final_ndc = mix(flat_ndc, trans_ndc, vec2<f32>(phase));

    var out: VertexOutput;
    out.position = vec4<f32>(final_ndc, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0) { discard; }

    // Back-face culling via UV winding
    let dudx = dpdx(in.uv);
    let dudy = dpdy(in.uv);
    if (dudx.x * dudy.y - dudx.y * dudy.x < 0.0) { discard; }

    let col = sampleImage(in.uv);
    return vec4<f32>(col.rgb * col.a, col.a);
}"""

    private const val HALF_PI = (Math.PI / 2.0).toFloat()
    private const val FOV = 4f
    private const val FACE_DEPTH = FOV / (FOV - 1f)
    private const val PHASE_IN_END = 0.1f
    private const val PHASE_OUT_START = 0.9f

    private fun mat4(
        m00: Float, m01: Float, m02: Float, m03: Float,
        m10: Float, m11: Float, m12: Float, m13: Float,
        m20: Float, m21: Float, m22: Float, m23: Float,
        m30: Float, m31: Float, m32: Float, m33: Float,
    ) = floatArrayOf(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    )

    private fun rotateY(angle: Float): FloatArray {
        val c = cos(angle)
        val s = sin(angle)
        return mat4(
            c, 0f, -s, 0f,
            0f, 1f, 0f, 0f,
            s, 0f, c, 0f,
            0f, 0f, 0f, 1f,
        )
    }

    private fun translate(x: Float, y: Float, z: Float) = mat4(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        x, y, z, 1f,
    )

    private fun scale(x: Float, y: Float, z: Float) = mat4(
        x, 0f, 0f, 0f,
        0f, y, 0f, 0f,
        0f, 0f, z, 0f,
        0f, 0f, 0f, 1f,
    )

    private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val result = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                result[col * 4 + row] = sum
            }
        }
        return result
    }

    private fun buildFaceMatrix(
        rotAngle: Float,
        screenAspect: Float,
        imgWidth: Float,
        imgHeight: Float,
        isSide: Boolean,
        phase: Float,
    ): FloatArray {
        val faceScaleMat = scale(FACE_DEPTH, (imgHeight / imgWidth) * screenAspect * FACE_DEPTH, 1f)
        val baseMat = if (isSide) {
            multiply(rotateY(HALF_PI), multiply(translate(0f, 0f, FACE_DEPTH), faceScaleMat))
        } else {
            multiply(translate(0f, 0f, FACE_DEPTH), faceScaleMat)
        }
        val worldMat = multiply(rotateY(-rotAngle), baseMat)
        // Perspective projection: output.w = z + fov, ndc = xy / w
        val projMat = mat4(
            FOV, 0f, 0f, 0f,
            0f, FOV, 0f, 0f,
            0f, 0f, 1f, 1f,
            0f, 0f, 0f, FOV,
        )
        val mat = multiply(projMat, worldMat)
        mat[14] = phase
        return mat
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
        val t = if (frac > 0f) frac else 1f + frac

        val frontPhase = if (t < PHASE_IN_END) t / PHASE_IN_END else 1f
        val sidePhase =
            if (t > PHASE_OUT_START) 1f - (t - PHASE_OUT_START) / (1f - PHASE_OUT_START) else 1f

        val rotAngle = when {
            t < PHASE_IN_END -> 0f
            t > PHASE_OUT_START -> HALF_PI
            else -> (t - PHASE_IN_END) / (PHASE_OUT_START - PHASE_IN_END) * HALF_PI
        }

        val screenAspect = dst.width.toFloat() / dst.height.toFloat()

        // When frac > 0: page1 is front (rotating away), page2 is side (rotating in)
        // When frac < 0: page2 is front, page1 is side
        val frontPage: ImagePage
        val sidePage: ImagePage
        if (frac > 0f) {
            frontPage = page1
            sidePage = page2
        } else {
            frontPage = page2
            sidePage = page1
        }

        val frontImg = frontPage.image
        val sideImg = sidePage.image

        val frontMat = buildFaceMatrix(
            rotAngle, screenAspect,
            frontImg?.width?.toFloat() ?: 1f,
            frontImg?.height?.toFloat() ?: 1f,
            isSide = false,
            phase = frontPhase,
        )
        val sideMat = buildFaceMatrix(
            rotAngle, screenAspect,
            sideImg?.width?.toFloat() ?: 1f,
            sideImg?.height?.toFloat() ?: 1f,
            isSide = true,
            phase = sidePhase,
        )

        // Render back-to-front for correct overlap
        if (t < 0.5f) {
            render(sidePage, encoder, dst, sideMat)
            render(frontPage, encoder, dst, frontMat)
        } else {
            render(frontPage, encoder, dst, frontMat)
            render(sidePage, encoder, dst, sideMat)
        }
    }

    internal fun render(
        page: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        matrix: FloatArray,
    ) {
        val image = page.image ?: return
        val res = image.prepareForRender(dst, page.x, page.y, page.scale) ?: return

        val byteBuffer = ByteBuffer.allocateDirect(96).apply {
            order(ByteOrder.nativeOrder())
            putFloat(0, res.x)
            putFloat(4, res.y)
            putFloat(8, res.scale)
            putFloat(12, res.mipmap.tilesize.toFloat())
            putFloat(16, res.mipmap.tilesCols.toFloat())
            putFloat(20, res.mipmap.tilesRows.toFloat())
            putFloat(24, dst.width.toFloat())
            putFloat(28, dst.height.toFloat())
            for (i in matrix.indices) {
                putFloat(32 + i * 4, matrix[i])
            }
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

        pass.draw(6144)
        pass.end()
    }
}
