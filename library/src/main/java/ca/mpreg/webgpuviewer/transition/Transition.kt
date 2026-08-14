package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder

abstract class Transition {
    open val code: String = ""

    protected val device get() = WebGpuRenderer.device

    protected open val pipeline: GPURenderPipeline by lazy {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        device.createRenderPipeline(
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

    abstract fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    )

    companion object {
        // Shared blit pipeline for all transitions
        private val blitPipeline: GPURenderPipeline by lazy {
            val device = WebGpuRenderer.device
            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(BLIT_SHADER))
            )
            device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                    fragment = GPUFragmentState(
                        shaderModule, entryPoint = "fs_main", targets = arrayOf(
                            GPUColorTargetState(
                                format = TextureFormat.RGBA8Unorm, blend = GPUBlendState(
                                    color = GPUBlendComponent(
                                        srcFactor = BlendFactor.One,
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

        private const val BLIT_SHADER = """
struct Uniforms {
    offset: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0)
    );
    
    let pos = positions[vertex_index];
    
    // Apply offset to position
    let offset_pos = pos + uniforms.offset;
    
    // Convert to NDC
    let ndc_x = offset_pos.x * 2.0 - 1.0;
    let ndc_y = 1.0 - offset_pos.y * 2.0;
    
    var out: VertexOutput;
    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(src_tex, src_sampler, in.uv);
}
"""

        private val blitSampler by lazy {
            WebGpuRenderer.device.createSampler()
        }

        private val blitByteBuffer = ThreadLocal.withInitial {
            ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder())
        }

        // Cache for current transition - 2 slots for page1 and page2
        private val cacheLock = Any()

        // Textures - always non-null once first render happens at a given size
        private var texture1: GPUTexture? = null
        private var texture2: GPUTexture? = null
        private var view1: GPUTextureView? = null
        private var view2: GPUTextureView? = null

        // Cache validity tracking
        private var cachedPage1: ImagePage? = null
        private var cachedPage2: ImagePage? = null
        private var cachedX1 = 0f
        private var cachedY1 = 0f
        private var cachedScale1 = 0f
        private var cachedFrameVersion1 = -1
        private var cachedX2 = 0f
        private var cachedY2 = 0f
        private var cachedScale2 = 0f
        private var cachedFrameVersion2 = -1

        private var cacheWidth = 0
        private var cacheHeight = 0

        // Textures pending destruction (deferred to avoid use-after-free)
        private var pendingDestroy1: GPUTexture? = null
        private var pendingDestroy2: GPUTexture? = null

        private fun ensureTexturesLocked(width: Int, height: Int) {
            // Destroy old pending textures (safe now - at least one frame has passed)
            pendingDestroy1?.destroy()
            pendingDestroy2?.destroy()
            pendingDestroy1 = null
            pendingDestroy2 = null

            // Recreate if size changed
            if (cacheWidth != width || cacheHeight != height) {
                // Defer destruction of old textures
                pendingDestroy1 = texture1
                pendingDestroy2 = texture2

                // Create new textures
                texture1 = WebGpuRenderer.device.createTexture(
                    GPUTextureDescriptor(
                        size = GPUExtent3D(width, height),
                        format = TextureFormat.RGBA8Unorm,
                        usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding
                    )
                )
                texture2 = WebGpuRenderer.device.createTexture(
                    GPUTextureDescriptor(
                        size = GPUExtent3D(width, height),
                        format = TextureFormat.RGBA8Unorm,
                        usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding
                    )
                )
                view1 = texture1!!.createView()
                view2 = texture2!!.createView()

                // Invalidate cache
                cachedPage1 = null
                cachedPage2 = null
                cacheWidth = width
                cacheHeight = height
            }
        }

        /**
         * Invalidate the transition cache. Call when transition ends.
         * Keeps textures allocated for reuse.
         */
        fun invalidateCache() {
            synchronized(cacheLock) {
                cachedPage1 = null
                cachedPage2 = null
            }
        }

        /**
         * Get cached texture view for a page, rendering if needed.
         * Returns null only if the page has no images.
         */
        internal fun getCachedTexture(
            page: ImagePage,
            isPage1: Boolean,
            encoder: GPUCommandEncoder,
            dstWidth: Int,
            dstHeight: Int,
            renderPage: (GPUCommandEncoder, GPUTexture) -> Unit
        ): GPUTextureView? {
            if (page.images.all { it == null }) return null

            val pageX = page.x
            val pageY = page.y
            val pageScale = page.scale
            val pageFrameVersion = page.frameVersion

            // Check cache validity and ensure textures exist (lock only for metadata).
            // GPU command recording does not need the lock - it always runs on the single
            // GPU render thread, and cacheLock only guards against invalidateCache() from UI thread.
            val (texture, view, needsRender) = synchronized(cacheLock) {
                ensureTexturesLocked(dstWidth, dstHeight)

                val texture = if (isPage1) texture1!! else texture2!!
                val view = if (isPage1) view1!! else view2!!
                val cachedPage = if (isPage1) cachedPage1 else cachedPage2
                val cachedX = if (isPage1) cachedX1 else cachedX2
                val cachedY = if (isPage1) cachedY1 else cachedY2
                val cachedScale = if (isPage1) cachedScale1 else cachedScale2
                val cachedFrame = if (isPage1) cachedFrameVersion1 else cachedFrameVersion2

                val cacheHit =
                    cachedPage === page && cachedX == pageX && cachedY == pageY && cachedScale == pageScale && cachedFrame == pageFrameVersion

                Triple(texture, view, !cacheHit)
            }

            if (!needsRender) return view

            // Record GPU commands outside the lock - GPU thread is single-threaded
            encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = view,
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            ).end()

            renderPage(encoder, texture)

            // Update cache metadata
            synchronized(cacheLock) {
                if (isPage1) {
                    cachedPage1 = page
                    cachedX1 = pageX
                    cachedY1 = pageY
                    cachedScale1 = pageScale
                    cachedFrameVersion1 = pageFrameVersion
                } else {
                    cachedPage2 = page
                    cachedX2 = pageX
                    cachedY2 = pageY
                    cachedScale2 = pageScale
                    cachedFrameVersion2 = pageFrameVersion
                }
            }

            return view
        }

        /**
         * Blit a cached texture to the destination with an offset.
         * Does nothing if cachedView is null.
         */
        internal fun blitCached(
            encoder: GPUCommandEncoder,
            dst: GPUTexture,
            cachedView: GPUTextureView?,
            offsetX: Float,
            offsetY: Float,
            clearFirst: Boolean = false
        ) {
            if (cachedView == null) return

            val byteBuffer = blitByteBuffer.get()
            byteBuffer.clear()
            byteBuffer.putFloat(offsetX)
            byteBuffer.putFloat(offsetY)
            byteBuffer.flip()

            val uniformBuffer = WebGpuRenderer.device.createBuffer(
                GPUBufferDescriptor(size = 8, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
            )
            WebGpuRenderer.device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = dst.createView(),
                            loadOp = if (clearFirst) LoadOp.Clear else LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )

            pass.setPipeline(blitPipeline)
            pass.setBindGroup(
                0, WebGpuRenderer.device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = blitPipeline.getBindGroupLayout(0), entries = arrayOf(
                            GPUBindGroupEntry(0, buffer = uniformBuffer),
                            GPUBindGroupEntry(1, textureView = cachedView),
                            GPUBindGroupEntry(2, sampler = blitSampler)
                        )
                    )
                )
            )
            pass.draw(6)
            pass.end()
        }
    }
}
