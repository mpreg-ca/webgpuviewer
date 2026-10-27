package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTextureView
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Resizes [TilesFilterChain]'s filter output to the tile, in linear light. `resize(start, ratio)` is one
 * destination pixel's footprint in source texels: top-left and side. Under 1 Catmull-Rom enlarges; over 1
 * a box average shrinks. Reads through `load_linear(texel)`, which the source prepended to it supplies.
 */
private const val MATH = """
fn to_linear(c: vec4<f32>) -> vec4<f32> {
    let lo = c.rgb / 12.92;
    let hi = pow((max(c.rgb, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return vec4<f32>(select(hi, lo, c.rgb <= vec3<f32>(0.04045)), c.a);
}

fn to_srgb(c: vec4<f32>) -> vec4<f32> {
    let rgb = max(c.rgb, vec3<f32>(0.0));
    let lo = rgb * 12.92;
    let hi = 1.055 * pow(rgb, vec3<f32>(1.0 / 2.4)) - 0.055;
    return clamp(vec4<f32>(select(hi, lo, rgb <= vec3<f32>(0.0031308)), c.a), vec4<f32>(0.0), vec4<f32>(1.0));
}

fn catmull_rom_weights(t: f32) -> array<f32, 4> {
    let t2 = t * t;
    let t3 = t2 * t;

    return array<f32, 4>(
        -0.5 * t3 + t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}

fn catmull_rom(pos: vec2<f32>) -> vec4<f32> {
    let pixel = pos - 0.5;
    let base = vec2<i32>(floor(pixel));
    let f = fract(pixel);
    var wx = catmull_rom_weights(f.x);
    var wy = catmull_rom_weights(f.y);

    var sum = vec4<f32>(0.0);
    for (var j: i32 = 0; j < 4; j++) {
        var row = vec4<f32>(0.0);
        for (var i: i32 = 0; i < 4; i++) {
            row += load_linear(base + vec2<i32>(i - 1, j - 1)) * wx[i];
        }
        sum += row * wy[j];
    }
    return to_srgb(sum);
}

fn box_average(start: vec2<f32>, size: f32) -> vec4<f32> {
    let end = start + vec2<f32>(size);
    let lo = vec2<i32>(floor(start));
    let hi = vec2<i32>(ceil(end));

    var sum = vec4<f32>(0.0);
    var total = 0.0;
    for (var y: i32 = lo.y; y < hi.y; y++) {
        let wy = max(0.0, min(f32(y + 1), end.y) - max(f32(y), start.y));
        for (var x: i32 = lo.x; x < hi.x; x++) {
            let w = wy * max(0.0, min(f32(x + 1), end.x) - max(f32(x), start.x));
            sum += load_linear(vec2<i32>(x, y)) * w;
            total += w;
        }
    }
    return to_srgb(sum / max(total, 0.0001));
}

fn resize(start: vec2<f32>, ratio: f32) -> vec4<f32> {
    if (ratio > 1.0) {
        return box_average(start, ratio);
    }
    return catmull_rom(start + vec2<f32>(0.5 * ratio));
}
"""

/** [MATH] over the filters' output texture. */
private const val TEXTURE = """
struct Resize {
    offset: vec2<f32>,
    ratio: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
var<immediate> params: Resize;

fn load_linear(p: vec2<i32>) -> vec4<f32> {
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    return to_linear(textureLoad(src, clamp(p, vec2<i32>(0), last), 0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = floor(in.position.xy);
    return resize(params.offset + p * params.ratio, params.ratio);
}
""" + MATH

/**
 * [TilesFilterChain]'s last step with filters: draws each tile pixel from their output, cutting off the
 * halo since the footprint starts past it. A render pass, not a copy, as the filters' format differs from
 * the atlas's; pipelines are keyed by the atlas format.
 */
internal class TileResize {
    private companion object {
        // Shared by every instance, so never closed.
        val pipelines =
            FormatKeyed { format -> Fullscreen.buildPipeline(TEXTURE, format, "tile resize") }
    }

    private var group: GPUBindGroup? = null
    private var groupSource: GPUTextureView? = null
    private var groupFormat = 0

    private val bytes = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())

    /** [offsetX]/[offsetY]: the tile's start in [source] texels; [ratio]: [source] texels per tile pixel. */
    fun draw(
        pass: GPURenderPassEncoder,
        format: Int,
        source: GPUTextureView,
        offsetX: Float,
        offsetY: Float,
        ratio: Float,
    ) {
        val device = WebGpuRenderer.device
        bytes.clear()
        bytes.putFloat(offsetX).putFloat(offsetY).putFloat(ratio).putFloat(0f)
        bytes.flip()

        val pipeline = pipelines[format]
        var bound = group
        if (bound == null || groupSource !== source || groupFormat != format) {
            bound?.close()
            bound = device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.groupLayout(), label = "tile resize", entries = arrayOf(
                        GPUBindGroupEntry(0, textureView = source),
                    )
                )
            )
            group = bound
            groupSource = source
            groupFormat = format
        }
        pass.setPipeline(pipeline)
        pass.setImmediates(0, bytes)
        pass.setBindGroup(0, bound)
        pass.draw(3)
    }

    /** Drop everything without freeing it, for a lost device's objects. */
    fun forget() {
        drop()
        pipelines.built.forEach { forgetGroupLayout(it) }
        pipelines.clear()
    }

    fun cleanup() {
        group?.close()
        drop()
    }

    private fun drop() {
        group = null
        groupSource = null
        groupFormat = 0
    }
}
