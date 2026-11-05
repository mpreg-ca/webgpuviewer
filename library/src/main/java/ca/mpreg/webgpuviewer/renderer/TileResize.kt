package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * [Sampler] host for the filters' output: the bounds over the whole texture, halo included, and a
 * fragment stage drawing `resolve`'s colour clamped to [0, 1].
 */
private const val HOST = """
struct Resize {
    offset: vec2<f32>,
    ratio: f32,
}

@group(0) @binding(0) var resize_src: texture_2d<f32>;
var<immediate> resize_params: Resize;

fn load(p: vec2<i32>) -> vec4<f32> {
    return textureLoad(resize_src, p, 0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ratio = vec2<f32>(resize_params.ratio);
    let start = resize_params.offset + floor(in.position.xy) * ratio;
    let bounds = vec4<i32>(vec2<i32>(0), vec2<i32>(textureDimensions(resize_src)) - 1);
    return clamp(resolve(start, ratio, bounds), vec4<f32>(0.0), vec4<f32>(1.0));
}
"""

/**
 * [TilesFilterChain]'s last step with filters: draws each tile pixel from their output through a
 * [Sampler], cutting off the halo since the footprint starts past it. A render pass, not a copy, as
 * the filters' format differs from the atlas's.
 */
internal class TileResize {
    companion object {
        // Shared by every instance, so never closed.
        private val pipelines = HashMap<String, FormatKeyed<GPURenderPipeline>>()

        private fun pipeline(code: String, format: Int) = pipelines.getOrPut(code) {
            FormatKeyed { Fullscreen.buildPipeline(HOST + code, it, "tile resize") }
        }[format]

        /** Uncached, for [Sampler.validate]. Render thread. */
        fun buildResolve(code: String) {
            Fullscreen.buildPipeline(HOST + code, TextureFormat.RGBA8Unorm, "tile resize").close()
        }
    }

    private var group: GPUBindGroup? = null
    private var groupSource: GPUTextureView? = null

    // A bind group fits only its own pipeline's layout.
    private var groupPipeline: GPURenderPipeline? = null

    private val bytes = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())

    /** [offsetX]/[offsetY]: the tile's start in [source] texels; [ratio]: [source] texels per tile pixel. */
    fun draw(
        pass: GPURenderPassEncoder,
        format: Int,
        source: GPUTextureView,
        offsetX: Float,
        offsetY: Float,
        ratio: Float,
        sampler: Sampler,
    ) {
        val device = WebGpuRenderer.device
        bytes.clear()
        bytes.putFloat(offsetX).putFloat(offsetY).putFloat(ratio).putFloat(0f)
        bytes.flip()

        val pipeline = pipeline(sampler.code, format)
        var bound = group
        if (bound == null || groupSource !== source || groupPipeline !== pipeline) {
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
            groupPipeline = pipeline
        }
        pass.setPipeline(pipeline)
        pass.setImmediates(0, bytes)
        pass.setBindGroup(0, bound)
        pass.draw(3)
    }

    /** Drop everything without freeing it, for a lost device's objects. */
    fun forget() {
        drop()
        pipelines.values.forEach { byFormat -> byFormat.built.forEach { forgetGroupLayout(it) } }
        pipelines.clear()
    }

    fun cleanup() {
        group?.close()
        drop()
    }

    private fun drop() {
        group = null
        groupSource = null
        groupPipeline = null
    }
}
