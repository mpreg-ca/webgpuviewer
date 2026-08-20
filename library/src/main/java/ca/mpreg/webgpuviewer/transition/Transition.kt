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
import androidx.webgpu.GPURenderPassEncoder
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
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.transition.Transition.Companion.blitCached
import ca.mpreg.webgpuviewer.transition.Transition.Companion.cacheLock
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.transition.Transition.Companion.invalidateCache
import ca.mpreg.webgpuviewer.transition.Transition.Companion.isCached
import ca.mpreg.webgpuviewer.viewer.ImagePage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow

abstract class Transition {
    open val code: String = ""

    protected val device get() = WebGpuRenderer.device

    /**
     * True when [code]'s fragment stage already returns premultiplied alpha.
     *
     * A transition that samples a cached page texture is in that position: the cache was written
     * premultiplied, so re-multiplying by alpha on the way out would darken every edge. Such a
     * shader needs `One` for the colour source factor, the way [blitCached] does, rather than the
     * `SrcAlpha` that suits a shader resolving straight-alpha texels.
     */
    protected open val premultipliedOutput: Boolean = false

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
                                    srcFactor = if (premultipliedOutput) BlendFactor.One
                                    else BlendFactor.SrcAlpha,
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

    internal abstract fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
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
         * The rect [page]'s flat render occupies inside its cached texture, as normalised
         * (x1, y1, x2, y2) surface coordinates, or null if the page has nothing to draw. Mirrors
         * [getCachedTexture]'s own placement, so a warp maps the page's actual rect rather than
         * treating it as screen-shaped.
         */
        internal fun pageRect(page: ImagePage, dst: GPUTexture): FloatArray? {
            val image = page.images.firstOrNull() ?: return null
            if (image.mipmaps.isEmpty()) return null
            return image.placement(dst, page.x, page.y, page.scale)
        }

        private fun srgbToLinear(c: Float): Float =
            if (c <= 0.04045f) c / 12.92f else ((c + 0.055f) / 1.055f).pow(2.4f)

        private fun linearToSrgb(c: Float): Float =
            if (c <= 0.0031308f) c * 12.92f else 1.055f * c.pow(1f / 2.4f) - 0.055f

        /**
         * Blend [bg1] toward [bg2] by [t] in linear space - 50% between white and black should be
         * linear grey, not the lighter result a straight sRGB-byte lerp gives. [TransitionFade]'s
         * own shader mix matches this rate: it un-premultiplies before converting to linear and
         * re-premultiplies after converting back, rather than giving up on linear blending - so
         * both stay at the same perceptual pace without either one needing to give up correctness.
         */
        internal fun blendBackgroundColor(bg1: Int, bg2: Int, t: Float): Int {
            fun channel(shift: Int): Int {
                val c1 = srgbToLinear(((bg1 shr shift) and 0xFF) / 255f)
                val c2 = srgbToLinear(((bg2 shr shift) and 0xFF) / 255f)
                val blended = linearToSrgb(c1 + (c2 - c1) * t)
                return (blended * 255f).toInt().coerceIn(0, 255)
            }
            return 0xFF000000.toInt() or (channel(16) shl 16) or (channel(8) shl 8) or channel(0)
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
         * Called once a page turn settles on [newCurrentPage]. Slot 2 is often already a valid
         * render of it - prewarmed by [ImageViewerState] while it was still the *next* page - so
         * this swaps it into slot 1 instead of discarding it. Falls back to a full wipe (like the
         * unconditional [invalidateCache] this replaces) when neither slot matches.
         */
        fun rotateCacheOnPageChange(newCurrentPage: ImagePage) {
            synchronized(cacheLock) {
                when {
                    cacheHitLocked(newCurrentPage, true) -> cachedPage2 = null
                    cacheHitLocked(newCurrentPage, false) -> {
                        val t = texture1; texture1 = texture2; texture2 = t
                        val v = view1; view1 = view2; view2 = v
                        cachedPage1 = cachedPage2
                        cachedX1 = cachedX2
                        cachedY1 = cachedY2
                        cachedScale1 = cachedScale2
                        cachedFrameVersion1 = cachedFrameVersion2
                        cachedPage2 = null
                    }

                    else -> {
                        cachedPage1 = null
                        cachedPage2 = null
                    }
                }
            }
        }

        /** Must hold [cacheLock]. Shared by [getCachedTexture] and [isCached]. */
        private fun cacheHitLocked(page: ImagePage, isPage1: Boolean): Boolean {
            val cachedPage = if (isPage1) cachedPage1 else cachedPage2
            val cachedX = if (isPage1) cachedX1 else cachedX2
            val cachedY = if (isPage1) cachedY1 else cachedY2
            val cachedScale = if (isPage1) cachedScale1 else cachedScale2
            val cachedFrame = if (isPage1) cachedFrameVersion1 else cachedFrameVersion2
            return cachedPage === page && cachedX == page.x && cachedY == page.y &&
                    cachedScale == page.scale && cachedFrame == page.frameVersion
        }

        /**
         * True if [getCachedTexture] would return a cache hit for [page] right now, with no
         * rendering - lets a caller (the shared background worker) decide whether warming this
         * page is still necessary before spending a turn on it.
         */
        internal fun isCached(page: ImagePage, isPage1: Boolean): Boolean =
            synchronized(cacheLock) { cacheHitLocked(page, isPage1) }

        /**
         * Get cached texture view for a page, rendering if needed. [renderPage] returns whether it
         * actually rendered - false leaves the texture cleared but doesn't mark it valid, so the
         * caller retries later rather than the cache reporting a hit for a blank frame. Returns
         * null if the page has no images or a needed render didn't happen.
         */
        internal fun getCachedTexture(
            page: ImagePage,
            isPage1: Boolean,
            encoder: GPUCommandEncoder,
            dstWidth: Int,
            dstHeight: Int,
            renderPage: (GPURenderPassEncoder, GPUTexture) -> Boolean
        ): GPUTextureView? {
            if (page.destroyed || page.images.all { it == null }) return null

            // Lock only for metadata - GPU recording runs on the single GPU thread and doesn't
            // need it; cacheLock only guards against invalidateCache() from the UI thread.
            val (texture, view, needsRender) = synchronized(cacheLock) {
                ensureTexturesLocked(dstWidth, dstHeight)
                val texture = if (isPage1) texture1!! else texture2!!
                val view = if (isPage1) view1!! else view2!!
                Triple(texture, view, !cacheHitLocked(page, isPage1))
            }

            if (!needsRender) return view

            val pageX = page.x
            val pageY = page.y
            val pageScale = page.scale
            val pageFrameVersion = page.frameVersion

            // Record outside the lock - GPU thread is single-threaded. The clear and the page's
            // draws share one pass, opened here so the callback can just draw into it.
            val pass = encoder.beginRenderPass(
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
            )
            val rendered = try {
                renderPage(pass, texture)
            } finally {
                pass.end()
            }

            if (!rendered) return null

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
         * Blit a cached texture to the destination with an offset. If [cachedView] is null, draws
         * nothing but still clears when [clearFirst] is set, so an undecoded page never leaves
         * stale swapchain content on screen.
         */
        internal fun blitCached(
            encoder: GPUCommandEncoder,
            dst: GPUTexture,
            cachedView: GPUTextureView?,
            offsetX: Float,
            offsetY: Float,
            clearFirst: Boolean = false
        ) {
            if (cachedView == null) {
                if (clearFirst) {
                    val pass = encoder.beginRenderPass(
                        GPURenderPassDescriptor(
                            colorAttachments = arrayOf(
                                GPURenderPassColorAttachment(
                                    view = dst.createView(),
                                    loadOp = LoadOp.Clear,
                                    storeOp = StoreOp.Store,
                                    clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                                )
                            )
                        )
                    )
                    pass.end()
                }
                return
            }

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
