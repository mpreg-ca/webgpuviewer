package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.filter.FilterFullscreen.Companion.VERTEX
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer

/**
 * A [Filter] that is one fragment pass over the whole frame - the shape every per-pixel filter
 * takes, and the one that can write the swapchain directly.
 *
 * Subclasses supply [code] (a `fs_main` taking [VERTEX]'s `VertexOutput`) and [entries] (its
 * group 0 bindings). Bind groups are kept per source texture rather than for the last one only,
 * since [FilterChain] rotates its textures across frames and a single-entry cache would then
 * miss every frame; a filter whose own bindings change - a new LUT, say - calls [rebind].
 *
 * There is no blend state: the pass covers every pixel and replaces it, and the frame it reads
 * already carries the alpha the swapchain needs.
 */
abstract class FilterFullscreen : Filter() {

    /** WGSL fragment stage. [VERTEX] is prepended, so `VertexOutput` and `in.uv` are in scope. */
    protected abstract val code: String

    protected open val pipeline: GPURenderPipeline by lazy {
        buildPipeline(code, outputFormat, label)
    }

    /** Group 0 bindings for this pass, with the chain's current input as [src]. */
    protected abstract fun entries(src: GPUTextureView): Array<GPUBindGroupEntry>

    // Keyed by the view's native handle, which outlives the wrapper object. Zero is free: a
    // live handle is never null.
    private val boundHandles = LongArray(CACHED_BIND_GROUPS)
    private val bindGroups = arrayOfNulls<GPUBindGroup>(CACHED_BIND_GROUPS)
    private var nextBindGroup = 0

    /** Drop the cached bind groups, for a filter whose own bindings have changed. */
    protected fun rebind() {
        boundHandles.fill(0L)
        bindGroups.fill(null)
        invalidate()
    }

    private fun bindGroupFor(src: GPUTextureView): GPUBindGroup {
        val handle = src.handle
        for (i in boundHandles.indices) {
            if (boundHandles[i] == handle) bindGroups[i]?.let { return it }
        }

        val group = device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.getBindGroupLayout(0), label = label, entries = entries(src)
            )
        )
        boundHandles[nextBindGroup] = handle
        bindGroups[nextBindGroup] = group
        nextBindGroup = (nextBindGroup + 1) % CACHED_BIND_GROUPS
        return group
    }

    /** Prepare GPU state for this frame - uploads and the like, before the pass opens. */
    protected open fun prepare(srcWidth: Int, srcHeight: Int) {}

    override fun run(
        chain: FilterChain,
        encoder: GPUCommandEncoder,
        src: GPUTextureView,
        srcWidth: Int,
        srcHeight: Int,
        dst: GPUTextureView,
        dstWidth: Int,
        dstHeight: Int,
    ) {
        prepare(srcWidth, srcHeight)

        // Before the pass opens: prepare() may have replaced a binding and dropped the cache.
        val group = bindGroupFor(src)

        val pass = beginPass(encoder, dst, label)
        try {
            pass.setPipeline(pipeline)
            pass.setBindGroup(0, group)
            pass.draw(3)
        } finally {
            pass.end()
        }
    }

    companion object {
        /** Enough for [FilterChain]'s texture ring, so a steady chain builds none per frame. */
        private const val CACHED_BIND_GROUPS = 4

        /**
         * One triangle covering the viewport, with [0,1] uv running top-left to bottom-right -
         * the same orientation the viewer's own draws use, so a filter reads the frame upright.
         */
        const val VERTEX = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}
"""

        /** Pipeline for [VERTEX] plus [code], writing [format] with no blending. */
        fun buildPipeline(code: String, format: Int, label: String): GPURenderPipeline {
            val device = WebGpuRenderer.device
            val module = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(VERTEX + code))
            )
            return device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    label = label,
                    vertex = GPUVertexState(module, entryPoint = "vs_main"),
                    fragment = GPUFragmentState(
                        module, entryPoint = "fs_main",
                        targets = arrayOf(GPUColorTargetState(format = format))
                    ),
                    primitive = GPUPrimitiveState(topology = TriangleList),
                )
            )
        }

        /**
         * A pass over the whole of [dst]. Clears rather than loads: the triangle covers every
         * pixel, so loading would only cost an attachment read for nothing.
         */
        fun beginPass(
            encoder: GPUCommandEncoder, dst: GPUTextureView, label: String
        ): GPURenderPassEncoder = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                label = label, colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst,
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )
    }
}
