package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.BufferUsage
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILES_PER_BATCH
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE
import ca.mpreg.webgpuviewer.viewer.ImagePage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * A cache of the filtered render, cut into [TILE_SIZE]-square screen-resolution tiles.
 *
 * [RenderPage.render]'s box/Catmull-Rom filter is too expensive to run over the whole screen
 * every frame, and [RenderPage.renderFast] gives up sharpness for that speed. This sits between:
 * each frame draws the fast path first, then blits whatever filtered tiles exist on top, while a
 * background worker fills in the missing ones a few at a time. The view converges to filtered
 * quality within a handful of frames of going quiet, and never stalls to get there.
 *
 * Tiles live in *content space*: tile (tx, ty) holds the [TILE_SIZE] square of the image's
 * filtered rendering whose top-left sits tx*[TILE_SIZE] pixels right and ty*[TILE_SIZE] pixels
 * below the image's centre, at the current scale. Panning only moves where a tile lands on
 * screen, so the cache survives it untouched - which is also why generation can keep running
 * *during* a pan, with [draw] re-prioritising towards whatever just scrolled into view. Changing
 * scale is what invalidates: every tile of that image is dropped, and nothing is enqueued again
 * until the scale holds for two consecutive frames, so a pinch or zoom animation doesn't churn
 * out tiles it will immediately throw away.
 *
 * The blit layer is snapped as a whole to the nearest screen pixel. The grid is rigid, so the
 * snap can't open seams between tiles, and it keeps every blit an exact 1:1 texel copy - a tile
 * drawn at a fractional offset would be resampled and lose exactly the sharpness it exists to
 * provide. The at-most-half-pixel displacement against the fast underlay is invisible - but each
 * image snaps *independently*, so two abutting images can land up to a whole pixel apart, which
 * would show as a gap or an overlap right on their shared edge. So blits are clipped one pixel
 * inside the image's own rect: the outermost pixel band of every image always comes from the
 * fast underlay, which draws at exact unsnapped positions, and image boundaries meet exactly as
 * they would without tiles.
 *
 * Blitting is built to be nearly free per frame. Each tile owns its bind group and a tiny
 * uniform holding its grid coordinate, both created once at generation, so a blit encodes as
 * just setBindGroup + draw. Everything that changes per frame - the snapped anchor and the clip
 * rect - lives in one 32-byte per-image uniform, written only on frames where the image actually
 * moved, and the clip intersection runs in the vertex shader by clamping the quad's corners.
 *
 * Generation runs on the render thread but outside the render mutex, in batches of
 * [TILES_PER_BATCH] with a yield between batches, so a queued frame always gets the thread back
 * after a bounded amount of encoding. The tile size bounds how much GPU work each pass submits,
 * so a frame queued behind a batch is never waiting on one huge filtered draw. Pending tiles are
 * picked by priority at pull time: on-screen before the margin ring, centre-out within it,
 * against the viewport as of the most recent frame.
 *
 * Everything here - maps, queues, textures - is touched only on the render thread, which is
 * single-threaded. [cleanup] may be called from any thread and posts its work there.
 */
internal class TileRenderer(private val invalidate: () -> Unit) {
    companion object {
        const val TILE_SIZE = 256
        private const val TILES_PER_BATCH = 1

        /** Drop an image's tiles after this many rendered frames without it being drawn. */
        private const val KEEP_FRAMES = 600L
        private const val TAG = "TileRenderer"
    }

    /** Upper bound on cached tiles across all images, ~[maxTiles] * 256KB of texture memory. */
    var maxTiles = 192

    private val device get() = WebGpuRenderer.device

    private var frame = 0L
    private var workerActive = false
    private val workerScope = CoroutineScope(WebGpuRenderer.dispatcher + SupervisorJob())

    private class Tile(
        val texture: GPUTexture, val uniform: GPUBuffer, val bindGroup: GPUBindGroup
    ) {
        var lastUsed = 0L

        fun destroy() {
            bindGroup.close()
            uniform.destroy()
            texture.destroy()
        }
    }

    /** [page] is the page the image was last drawn under, for its destroyed flag. */
    private class ImageTiles(var scale: Float, val frameUniform: GPUBuffer, var page: ImagePage) {
        val tiles = HashMap<Long, Tile>()
        val pending = HashSet<Long>()

        // Values the current frameUniform contents were derived from, so a frame where the image
        // didn't move skips the write entirely and encodes nothing but the blit draws.
        var writtenSnapX = Float.NaN
        var writtenSnapY = Float.NaN
        var writtenDstW = Float.NaN
        var writtenDstH = Float.NaN
        var writtenHalfW = Float.NaN
        var writtenHalfH = Float.NaN

        /** True once the scale has held for two consecutive frames; gates generation. */
        var stable = false
        var lastSeen = 0L

        // The strictly visible tile range as of the last draw, in tile coordinates. The worker
        // prioritises against it at pull time, so a pan mid-fill redirects generation without
        // touching the queue.
        var txMin = 0
        var txMax = -1
        var tyMin = 0
        var tyMax = -1
    }

    private class Request(val image: Image, val state: ImageTiles, val tx: Int, val ty: Int)

    private val images = HashMap<Image, ImageTiles>()

    private fun key(tx: Int, ty: Int) = (tx.toLong() shl 32) or (ty.toLong() and 0xFFFFFFFFL)

    // Thread-local ByteBuffer to avoid per-blit allocation
    private val byteBufferLocal = ThreadLocal.withInitial {
        ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
    }

    // Nearest: tiles are blitted 1:1 at integer pixel positions, so this is an exact copy.
    private val blitSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Nearest,
                minFilter = FilterMode.Nearest,
            )
        )
    }

    private val blitPipeline: GPURenderPipeline by lazy {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(BLIT_SHADER))
        )
        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(module = shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    module = shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            // Tiles hold RenderPage's output, which is premultiplied, so One
                            // rather than SrcAlpha.
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
                primitive = GPUPrimitiveState(topology = PrimitiveTopology.TriangleList)
            )
        )
    }

    /**
     * Advance the frame counter and drop tiles of destroyed or long-unseen pages.
     *
     * Called once at the top of every rendered frame, including frames that don't draw tiles
     * (e.g. during a page transition), so eviction keeps ticking.
     */
    fun newFrame() {
        frame++
        if (images.isEmpty()) return
        val it = images.iterator()
        while (it.hasNext()) {
            val st = it.next().value
            if (st.page.destroyed || frame - st.lastSeen > KEEP_FRAMES) {
                st.tiles.values.forEach { tile -> tile.destroy() }
                st.tiles.clear()
                st.pending.clear()
                st.frameUniform.destroy()
                it.remove()
            }
        }
    }

    /**
     * Blit the cached tiles for a whole page and enqueue the missing ones.
     *
     * Takes the same placement arguments as [RenderPage.render] and decomposes into per-image
     * calls the same way, minus the background - the fast underlay draws that live, so fades
     * tied to the page's current position (overscroll, under-min-scale) never go stale in a
     * cached tile.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) {
        page.images.forEach { image ->
            image ?: return@forEach
            val offsetX = when (image.position) {
                Image.Position.LEFT -> (-0.5f * image.width) / dst.width
                Image.Position.RIGHT -> (0.5f * image.width) / dst.width
                Image.Position.SINGLE -> 0f
            }
            draw(pass, page, image, dst, page.x + x + offsetX, page.y + y, page.scale * scale)
        }
    }

    /**
     * Blit the cached tiles for one image and enqueue the missing ones.
     *
     * [x], [y] and [scale] are exactly what the caller passes to [RenderPage.renderFast] for the
     * same image, so the tiles land pixel-for-pixel (within the half-pixel layer snap) on top of
     * the fast underlay.
     *
     * [ImagePage.Draw] pages are drawn into externally and animated pages swap images per frame;
     * a cache would hold yesterday's content for both, so they stay on the fast path.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        page: ImagePage,
        image: Image,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) {
        if (page.destroyed || page is ImagePage.Draw || page.isAnimated) return
        if (image.mipmaps.isEmpty()) return

        val st = images.getOrPut(image) {
            ImageTiles(
                scale, device.createBuffer(
                    GPUBufferDescriptor(
                        size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                    )
                ), page
            )
        }
        st.page = page
        st.lastSeen = frame
        if (st.scale != scale) {
            // Last frame's pass, the only one that can reference these, is already submitted;
            // Dawn keeps a destroyed texture alive until the command buffers using it retire.
            st.tiles.values.forEach { it.destroy() }
            st.tiles.clear()
            st.pending.clear()
            st.scale = scale
            st.stable = false
            // The next frame is what proves the scale has settled, but when this is the last
            // frame of a zoom animation no further frame is coming, and nothing would enqueue
            // tiles until the next interaction. Ask for one; while the scale is still moving
            // this just coalesces into the motion's own invalidations.
            invalidate()
        } else {
            st.stable = true
        }

        val ts = TILE_SIZE.toFloat()
        // Screen-pixel position of the image centre. Content space hangs off it: content pixel c
        // sits at screen pixel anchor + c, so tiles keyed by content coordinates survive panning.
        val anchorX = dst.width / 2f + scale * ((x + WebGpuRenderer.offsetX) * dst.width + image.x)
        val anchorY =
            dst.height / 2f + scale * ((y + WebGpuRenderer.offsetY) * dst.height + image.y)
        val halfW = scale * image.width / 2f
        val halfH = scale * image.height / 2f

        // Wanted region: the viewport plus a one-tile margin ring, clipped to the image extent.
        val wantL = max(-anchorX - ts, -halfW)
        val wantR = min(dst.width - anchorX + ts, halfW)
        val wantT = max(-anchorY - ts, -halfH)
        val wantB = min(dst.height - anchorY + ts, halfH)
        if (wantL >= wantR || wantT >= wantB) {
            st.pending.clear()
            return
        }

        val tx0 = floor(wantL / ts).toInt()
        val tx1 = ceil(wantR / ts).toInt() - 1
        val ty0 = floor(wantT / ts).toInt()
        val ty1 = ceil(wantB / ts).toInt() - 1

        st.txMin = floor(-anchorX / ts).toInt()
        st.txMax = ceil((dst.width - anchorX) / ts).toInt() - 1
        st.tyMin = floor(-anchorY / ts).toInt()
        st.tyMax = ceil((dst.height - anchorY) / ts).toInt() - 1

        // The whole layer snaps to the nearest pixel together, so blits stay 1:1 and seamless.
        val snapX = round(anchorX)
        val snapY = round(anchorY)

        // Blits stop one pixel short of the image's edges. Neighbouring images snap
        // independently, so their tile layers can shift up to a pixel against each other; the
        // underlay owns the perimeter band instead, and abutting images join exactly as they
        // would without tiles. Images too small for an interior aren't worth caching at all.
        val clipL = snapX - halfW + 1f
        val clipT = snapY - halfH + 1f
        val clipR = snapX + halfW - 1f
        val clipB = snapY + halfH - 1f
        if (clipL >= clipR || clipT >= clipB) {
            st.pending.clear()
            return
        }

        // Everything the blits need per frame goes in the image's one uniform, written only when
        // something moved - snap, clip and dst size are all functions of these six values. One
        // write per buffer per frame, so the writeBuffer-vs-submit ordering hazard that forces
        // Draw.rect to allocate fresh buffers doesn't apply.
        val dstW = dst.width.toFloat()
        val dstH = dst.height.toFloat()
        if (st.writtenSnapX != snapX || st.writtenSnapY != snapY ||
            st.writtenHalfW != halfW || st.writtenHalfH != halfH ||
            st.writtenDstW != dstW || st.writtenDstH != dstH
        ) {
            val byteBuffer = byteBufferLocal.get()
            byteBuffer.clear()
            byteBuffer.putFloat(snapX)
            byteBuffer.putFloat(snapY)
            byteBuffer.putFloat(dstW)
            byteBuffer.putFloat(dstH)
            byteBuffer.putFloat(clipL)
            byteBuffer.putFloat(clipT)
            byteBuffer.putFloat(clipR)
            byteBuffer.putFloat(clipB)
            byteBuffer.flip()
            device.queue.writeBuffer(st.frameUniform, 0, byteBuffer)
            st.writtenSnapX = snapX
            st.writtenSnapY = snapY
            st.writtenDstW = dstW
            st.writtenDstH = dstH
            st.writtenHalfW = halfW
            st.writtenHalfH = halfH
        }

        // What a tile has to intersect to produce any fragments.
        val visL = max(clipL, 0f)
        val visT = max(clipT, 0f)
        val visR = min(clipR, dstW)
        val visB = min(clipB, dstH)

        var pipelineSet = false
        val desired = HashSet<Long>()
        for (tyi in ty0..ty1) {
            for (txi in tx0..tx1) {
                val key = key(txi, tyi)
                desired.add(key)
                val tile = st.tiles[key]
                if (tile != null) {
                    tile.lastUsed = frame
                    val px = snapX + txi * ts
                    val py = snapY + tyi * ts
                    // Margin-ring tiles usually sit entirely offscreen or outside the clip;
                    // skip those rather than encode a draw that can't produce fragments.
                    if (px < visR && px + ts > visL && py < visB && py + ts > visT) {
                        if (!pipelineSet) {
                            pass.setPipeline(blitPipeline)
                            pipelineSet = true
                        }
                        pass.setBindGroup(0, tile.bindGroup)
                        pass.draw(6)
                    }
                } else if (st.stable) {
                    st.pending.add(key)
                }
            }
        }
        // Tiles that panned out of the wanted region stop being worth generating.
        st.pending.retainAll(desired)
        if (st.pending.isNotEmpty()) schedule()
    }

    /**
     * Start the generation worker if it isn't running.
     *
     * The worker shares the render thread but not the render mutex, so the yield between batches
     * actually lets a queued frame through - the same reasoning as [WebGpuRenderer.onDispatcher].
     */
    private fun schedule() {
        if (workerActive) return
        workerActive = true
        workerScope.launch {
            try {
                while (true) {
                    var generated = 0
                    while (generated < TILES_PER_BATCH) {
                        val req = nextRequest() ?: break
                        try {
                            generate(req)
                            generated++
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            Log.e(TAG, "Tile render failed", e)
                        }
                    }
                    if (generated == 0) break
                    invalidate()
                    yield()
                }
            } finally {
                workerActive = false
            }
        }
    }

    /** Pull the highest-priority pending tile: on-screen first, then centre-out. */
    private fun nextRequest(): Request? {
        var bestImage: Image? = null
        var bestState: ImageTiles? = null
        var bestKey = 0L
        var bestPriority = Float.MAX_VALUE
        for ((image, st) in images) {
            if (st.pending.isEmpty()) continue
            if (st.page.destroyed || !st.stable) {
                st.pending.clear()
                continue
            }
            val centerTx = (st.txMin + st.txMax) * 0.5f
            val centerTy = (st.tyMin + st.tyMax) * 0.5f
            for (key in st.pending) {
                val tx = (key shr 32).toInt()
                val ty = key.toInt()
                val outX = max(max(st.txMin - tx, tx - st.txMax), 0)
                val outY = max(max(st.tyMin - ty, ty - st.tyMax), 0)
                val cx = tx - centerTx
                val cy = ty - centerTy
                val priority = max(outX, outY) * 1e6f + cx * cx + cy * cy
                if (priority < bestPriority) {
                    bestPriority = priority
                    bestImage = image
                    bestState = st
                    bestKey = key
                }
            }
        }
        val st = bestState ?: return null
        st.pending.remove(bestKey)
        return Request(bestImage!!, st, (bestKey shr 32).toInt(), bestKey.toInt())
    }

    /** Render one tile with the filtered shader and store it. Runs outside any frame. */
    private fun generate(req: Request) {
        val st = req.state
        if (st.page.destroyed || !st.stable) return
        val key = key(req.tx, req.ty)
        if (st.tiles.containsKey(key)) return
        evict()

        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(TILE_SIZE, TILE_SIZE),
                usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding,
                format = TextureFormat.RGBA8Unorm
            )
        )
        // The tile's own uniform holds just its grid coordinate, which never changes, so it and
        // the bind group are created once here and a blit is nothing but setBindGroup + draw.
        // 32 bytes rather than the 8 the shader reads: writeBuffer writes the whole capacity of
        // the shared scratch ByteBuffer, and a write must fit in the destination.
        val uniform = device.createBuffer(
            GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        )
        try {
            val encoder = device.createCommandEncoder()
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = texture.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                // Place the image so its centre lands at tile pixel (-tx*ts, -ty*ts); content
                // pixel c then lands at tile pixel c - t*ts, i.e. the tile holds exactly the
                // content square [t*ts, (t+1)*ts). Solved from prepareForRender's placement:
                // centre = ts/2 + scale * (x + image.x/ts + offsetX) * ts.
                val ts = TILE_SIZE.toFloat()
                val s = st.scale
                val x =
                    (-req.tx * ts - ts / 2f) / (s * ts) - req.image.x / ts - WebGpuRenderer.offsetX
                val y =
                    (-req.ty * ts - ts / 2f) / (s * ts) - req.image.y / ts - WebGpuRenderer.offsetY
                RenderPage.render(pass, req.image, texture, x, y, s)
            } finally {
                pass.end()
            }
            device.queue.submit(arrayOf(encoder.finish()))

            val byteBuffer = byteBufferLocal.get()
            byteBuffer.clear()
            byteBuffer.putFloat(req.tx.toFloat())
            byteBuffer.putFloat(req.ty.toFloat())
            byteBuffer.flip()
            device.queue.writeBuffer(uniform, 0, byteBuffer)

            val bindGroup = device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = blitPipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = st.frameUniform),
                        GPUBindGroupEntry(1, buffer = uniform),
                        GPUBindGroupEntry(2, textureView = texture.createView()),
                        GPUBindGroupEntry(3, sampler = blitSampler),
                    )
                )
            )
            st.tiles[key] = Tile(texture, uniform, bindGroup).also { it.lastUsed = frame }
        } catch (e: Exception) {
            texture.destroy()
            uniform.destroy()
            throw e
        }
    }

    /**
     * Evict least-recently-used tiles down to [maxTiles], best-effort.
     *
     * Tiles used this frame or last are never touched: the pass that blitted them may not have
     * been submitted yet, and destroying a texture a recording encoder references is a
     * validation error. If everything is that fresh the cap is allowed to overshoot - the
     * wanted set is bounded by the viewport, so the overshoot is too.
     */
    private fun evict() {
        var total = 0
        for (st in images.values) total += st.tiles.size
        if (total < maxTiles) return

        val candidates = ArrayList<Triple<ImageTiles, Long, Tile>>()
        for (st in images.values) {
            for ((k, t) in st.tiles) {
                if (t.lastUsed < frame - 1) candidates.add(Triple(st, k, t))
            }
        }
        candidates.sortBy { it.third.lastUsed }
        var i = 0
        while (total >= maxTiles && i < candidates.size) {
            val (st, k, t) = candidates[i]
            st.tiles.remove(k)
            t.destroy()
            total--
            i++
        }
    }

    /**
     * Free every tile. Safe from any thread; the destruction runs on the render thread, where
     * the worker finds nothing pending and stops. Not a permanent shutdown: the surface can be
     * torn down and recreated with the same viewer state, and rendering simply refills the cache
     * - a dead tile layer after the app comes back to the foreground would be a silent quality
     * regression.
     */
    fun cleanup() {
        workerScope.launch {
            images.values.forEach { st ->
                st.tiles.values.forEach { it.destroy() }
                st.tiles.clear()
                st.pending.clear()
                st.frameUniform.destroy()
            }
            images.clear()
        }
    }
}

private const val BLIT_SHADER = """
const TS: f32 = $TILE_SIZE.0;

struct FrameParams {
    // Snapped screen-pixel position of the image centre; the tile grid hangs off it.
    snap: vec2<f32>,
    dst_size: vec2<f32>,
    // The image's rect inset by a pixel, in screen pixels - the underlay owns the perimeter.
    clip: vec4<f32>,
}

struct TileParams {
    t: vec2<f32>,  // this tile's grid coordinate, fixed for the tile's lifetime
}

@group(0) @binding(0) var<uniform> frame_params: FrameParams;
@group(0) @binding(1) var<uniform> tile_params: TileParams;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var src_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), // Top-left
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(1.0, 0.0), // Top-right
        vec2<f32>(0.0, 1.0), // Bottom-left
        vec2<f32>(1.0, 1.0)  // Bottom-right
    );

    let pos = corners[vertex_index];

    // Clamping the corners to the clip rect shrinks the quad and its uv window in step, so
    // whatever survives is still a 1:1 texel copy. Offscreen overhang is left to NDC clipping.
    let origin = frame_params.snap + tile_params.t * TS;
    let p = clamp(origin + pos * TS, frame_params.clip.xy, frame_params.clip.zw);

    var out: VertexOutput;
    out.position = vec4<f32>(
        p.x / frame_params.dst_size.x * 2.0 - 1.0,
        1.0 - p.y / frame_params.dst_size.y * 2.0,
        0.0, 1.0
    );
    out.uv = (p - origin) / TS;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1:1 at integer positions with a nearest sampler: an exact copy of the tile's texels,
    // already premultiplied by RenderPage.
    return textureSample(src_tex, src_sampler, in.uv);
}
"""
