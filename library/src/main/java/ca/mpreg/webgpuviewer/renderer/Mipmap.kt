package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.ErrorFilter
import androidx.webgpu.ErrorType
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUOrigin3D
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.min
import kotlin.time.Duration.Companion.milliseconds

class Mipmap(
    val width: Int,
    val height: Int,
    val scale: Float,
    val tilesCols: Int,
    val tilesRows: Int,
    val tilesize: Int,
    /**
     * Format of the tile textures. Follows the pixels this level was built from, never the
     * swapchain: an SDR image keeps 8-bit tiles even while HDR is on screen, since sampling one
     * into a float target needs no help and widening it would only cost bandwidth.
     */
    val format: Int,
) {
    /** Bytes one tile texel occupies, for the row strides [upload] and [update] copy at. */
    private val bytesPerPixel: Int
        get() = if (format == TextureFormat.RGBA16Float) 8 else 4

    companion object {
        private val device get() = WebGpuRenderer.device

        /**
         * Rough size of a single `writeTexture` call, in bytes.
         *
         * Big enough that per-call overhead stays noise, small enough that one copy fits in the
         * slack of a frame. A whole 2000x3000 page in one call is ~24MB of memcpy on the render
         * thread, which is several frames' worth.
         */
        private const val UPLOAD_CHUNK_BYTES = 1 shl 20

        /** Texel multiple every tile copy is widened to - see [writeSpan]. Two isn't enough. */
        private const val COPY_ALIGNMENT = 4

        /** [writeSpan]'s padded rows: one per uploading thread, grown, never shrunk. */
        private val padScratch = ThreadLocal<ByteBuffer>()

        /**
         * Build a mipmap level from [pixels] and upload it.
         *
         * Suspends between chunks, so it must run outside the render mutex (see
         * [WebGpuRenderer.onDispatcher]) for the yields to be worth anything. The level is only
         * returned once every chunk has landed, so no caller can sample a half-filled texture.
         *
         * With [checkOom], a tile the driver cannot allocate throws [TextureOutOfMemory] instead of
         * leaving an error texture behind - for oversized tiles a caller can retry smaller.
         */
        suspend fun create(
            pixels: ByteBuffer, width: Int, height: Int, scale: Float, tilesize: Int,
            format: Int = TextureFormat.RGBA8Unorm, checkOom: Boolean = false,
        ): Mipmap {
            val mipmap = Mipmap(
                width = width,
                height = height,
                scale = scale,
                tilesCols = ceil(width.toFloat() / tilesize).toInt(),
                tilesRows = ceil(height.toFloat() / tilesize).toInt(),
                tilesize = tilesize,
                format = format,
            )
            try {
                mipmap.upload(pixels, checkOom)
            } catch (e: Throwable) {
                // Yielding makes the upload cancellable, so a half-built level can now exist.
                // Free whatever landed before rethrowing - the caller never sees this instance
                // and so can't free it itself.
                mipmap.cleanup()
                throw e
            }
            return mipmap
        }
    }

    class TextureOutOfMemory(message: String) : Exception(message)

    /** Allocate the tile textures and copy [pixels] into them a chunk at a time. */
    private suspend fun upload(pixels: ByteBuffer, checkOom: Boolean) {
        val need = width.toLong() * height * bytesPerPixel
        require(pixels.capacity() >= need) { "pixels hold ${pixels.capacity()} B, ${width}x$height needs $need" }
        val rowsPerChunk = (UPLOAD_CHUNK_BYTES / (width * bytesPerPixel)).coerceAtLeast(1)

        for (r in 0 until tilesRows) {
            val tileHeight = min((r + 1) * tilesize, height) - (r * tilesize)
            val y = r * tilesize
            for (c in 0 until tilesCols) {
                val x = c * tilesize
                val tileWidth = min((c + 1) * tilesize, width) - (c * tilesize)

                Log.i("Renderer", "Create tile $c $r $tileWidth $tileHeight $x $y")

                // Unyielded driver work - not on the back of the chunk just uploaded.
                yield()
                if (checkOom) device.pushErrorScope(ErrorFilter.OutOfMemory)
                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        // Aligned, for [writeSpan]: up to three padding columns.
                        size = GPUExtent3D(alignedWidth(tileWidth), tileHeight),
                        format = format,
                        usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment,
                    )
                )
                textures.add(texture)
                if (checkOom && popErrorScopePumped() == ErrorType.OutOfMemory) {
                    throw TextureOutOfMemory("${tileWidth}x$tileHeight tile")
                }
                textureViews.add(texture.createView())

                var row = 0
                while (row < tileHeight) {
                    val rows = min(rowsPerChunk, tileHeight - row)
                    writeSpan(texture, pixels, x, tileWidth, x, x + tileWidth, y + row, row, rows)

                    row += rows
                    yield()
                }
            }
        }

        for (r in 0 until 2) {
            val row = r.coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + c.coerceAtMost(tilesCols - 1)
                tiles.add(textures[i])
                tileViews.add(textureViews[i])
            }
        }

        if (tilesCols <= 2 && tilesRows <= 2) {
            cachedQuad = Quad(tiles, tileViews, 0, 0)
        }
    }

    var textures: MutableList<GPUTexture> = mutableListOf()
    private var textureViews: MutableList<GPUTextureView> = mutableListOf()
    private var tiles: MutableList<GPUTexture> = mutableListOf()
    private var tileViews: MutableList<GPUTextureView> = mutableListOf()

    private var cachedQuad: Quad? = null

    constructor(texture: GPUTexture, scale: Float, tilesize: Int) : this(
        texture.width, texture.height, scale, 1, 1, tilesize, texture.format
    ) {
        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    // A drawable image is painted by the app from ARGB ints, so it is SDR by nature - the draw
    // pipelines read their format off the texture, so 8-bit here needs nothing else.
    constructor(width: Int, height: Int) : this(
        width, height, 1f, 1, 1, 4096, TextureFormat.RGBA8Unorm
    ) {
        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(width, height),
                format = format,
                usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment or TextureUsage.StorageBinding,
            )
        )

        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    /** popErrorScope only resolves while events are processed - pump them until it does. */
    private suspend fun popErrorScopePumped(): Int = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1.milliseconds)
            }
        }
        try {
            device.popErrorScope()
        } finally {
            pump.cancel()
        }
    }

    internal fun cleanup() {
        cachedQuad = null
        lastQuad = null
        lastQuadTX = -1
        lastQuadTY = -1
        // Every one of these is AutoCloseable over a Dawn handle with no finalizer behind it, so dropping the
        // reference alone leaks the native object: destroy() frees the memory, close() releases the handle.
        // tiles and tileViews only alias entries of textures and textureViews, so closing those twice would
        // release a handle that is already gone - they are just cleared.
        tileUniforms?.forEach { it?.destroyAndRelease() }
        tileUniforms = null
        textureViews.forEach { view -> view.close() }
        textureViews.clear()
        tileViews.clear()
        textures.forEach { tex -> tex.destroyAndRelease() }
        textures.clear()
        tiles.clear()
    }

    /**
     * Rewrites [rect] (default: all) from [pixels], a full image of this level, in yielding chunks
     * like [upload]; run via [WebGpuRenderer.onDispatcher]. False if cleaned up part way.
     */
    suspend fun update(pixels: ByteBuffer, rect: Rect? = null): Boolean {
        val need = width.toLong() * height * bytesPerPixel
        require(pixels.capacity() >= need) { "pixels hold ${pixels.capacity()} B, ${width}x$height needs $need" }
        val left = (rect?.left ?: 0).coerceIn(0, width)
        val top = (rect?.top ?: 0).coerceIn(0, height)
        val right = (rect?.right ?: width).coerceIn(left, width)
        val bottom = (rect?.bottom ?: height).coerceIn(top, height)
        val rowsPerChunk = (UPLOAD_CHUNK_BYTES / (width * bytesPerPixel)).coerceAtLeast(1)
        val tileCount = tilesRows * tilesCols

        for (r in top / tilesize until ceilDiv(bottom, tilesize)) {
            val tileY = r * tilesize
            val y0 = maxOf(top, tileY)
            val y1 = minOf(bottom, tileY + tilesize, height)
            for (c in left / tilesize until ceilDiv(right, tilesize)) {
                val tileX = c * tilesize
                val x0 = maxOf(left, tileX)
                val x1 = minOf(right, tileX + tilesize, width)
                var y = y0
                while (y < y1) {
                    // Cleanup can only land between chunks.
                    if (textures.size != tileCount) return false
                    val rows = min(rowsPerChunk, y1 - y)
                    writeSpan(
                        textures[r * tilesCols + c], pixels, tileX, min(tilesize, width - tileX),
                        x0, x1, y, y - tileY, rows,
                    )
                    y += rows
                    yield()
                }
            }
        }
        return textures.size == tileCount
    }

    private fun ceilDiv(a: Int, b: Int) = (a + b - 1) / b

    private fun alignedWidth(w: Int) = (w + COPY_ALIGNMENT - 1) and (COPY_ALIGNMENT - 1).inv()

    /**
     * Copies columns [x0, x1) of rows [y, y + rows) of [pixels], a full image of this level, into
     * [texture] - the tile at column [tileX], [tileWidth] wide - from its row [dstY].
     *
     * Every copy is a multiple of [COPY_ALIGNMENT] texels wide: a PowerVR Rogue driver lost the
     * device, as out-of-memory, on pages 1125 pixels wide with almost nothing allocated - and
     * still did padded to an even width, but not to a multiple of four. Padding only the texture,
     * or only the source rows' stride, with the copy at the real width, lost it too: the copy's
     * own width is what has to be aligned. So the span is widened to aligned columns,
     * and where that runs off the tile into its padding, the last real column is repeated there
     * from a packed copy of the rows.
     */
    private fun writeSpan(
        texture: GPUTexture, pixels: ByteBuffer, tileX: Int, tileWidth: Int,
        x0: Int, x1: Int, y: Int, dstY: Int, rows: Int,
    ) {
        val start = (x0 - tileX) and (COPY_ALIGNMENT - 1).inv()
        val end = start + alignedWidth(x1 - tileX - start)
        // Columns the image has; past them, only padding.
        val real = min(end, tileWidth)
        val spanBytes = (end - start) * bytesPerPixel

        val layout: GPUTexelCopyBufferLayout
        val data: ByteBuffer
        if (end == real) {
            data = pixels
            layout = GPUTexelCopyBufferLayout(
                // Long arithmetic: y * width overflows Int well before the byte offset does on
                // a large page.
                offset = (y.toLong() * width + tileX + start) * bytesPerPixel,
                bytesPerRow = width * bytesPerPixel,
                rowsPerImage = height,
            )
        } else {
            val need = spanBytes * rows
            data = padScratch.get()?.takeIf { it.capacity() >= need }
                ?: ByteBuffer.allocateDirect(need).also { padScratch.set(it) }
            data.clear()
            val realBytes = (real - start) * bytesPerPixel
            val src = pixels.duplicate()
            for (r in 0 until rows) {
                val rowStart = (((y + r).toLong() * width + tileX + start) * bytesPerPixel).toInt()
                src.limit(rowStart + realBytes).position(rowStart)
                data.put(src)
                // The padding columns: the last real pixel again.
                repeat(end - real) {
                    src.limit(rowStart + realBytes).position(rowStart + realBytes - bytesPerPixel)
                    data.put(src)
                }
            }
            data.flip()
            layout = GPUTexelCopyBufferLayout(offset = 0, bytesPerRow = spanBytes, rowsPerImage = rows)
        }

        device.queue.writeTexture(
            dataLayout = layout,
            data = data,
            destination = GPUTexelCopyTextureInfo(texture = texture, origin = GPUOrigin3D(x = start, y = dstY)),
            writeSize = GPUExtent3D(end - start, rows),
        )
    }

    class Quad(
        val tiles: List<GPUTexture>, val tileViews: List<GPUTextureView>, val x: Int, val y: Int
    )

    /**
     * One tile overlapping a queried rect, at its own pixel offset within the mipmap. [uniform] is
     * its own persistent placement buffer - see [tileUniforms] for why it needs one of its own.
     */
    class TileRect(
        val texture: GPUTexture,
        val view: GPUTextureView,
        val x: Int,
        val y: Int,
        val uniform: GPUBuffer
    )

    /**
     * One small uniform buffer per physical tile, created on first use and rewritten every frame
     * that tile is drawn - never shared between tiles. `writeBuffer` calls all land before any
     * command buffer submitted afterwards executes, so a shared buffer would let the last of
     * several tiles drawn in one frame win for all of them; one buffer per tile has no such race.
     */
    private var tileUniforms: Array<GPUBuffer?>? = null

    private fun tileUniformFor(index: Int): GPUBuffer {
        val arr = tileUniforms ?: arrayOfNulls<GPUBuffer>(textures.size).also { tileUniforms = it }
        return arr[index] ?: device.createBuffer(
            GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
        ).also { arr[index] = it }
    }

    /**
     * Every tile overlapping [left]..[right] by [top]..[bottom] (mipmap pixels) - unlike [getQuad]
     * which always hands back exactly 2x2, letting a caller draw each tile separately instead of
     * needing everything to fit one window. See [Image.prepareTilesForRender].
     */
    fun tilesInRect(left: Float, top: Float, right: Float, bottom: Float): List<TileRect> {
        val l = left.coerceIn(0f, width.toFloat())
        val t = top.coerceIn(0f, height.toFloat())
        val r = right.coerceIn(0f, width.toFloat())
        val b = bottom.coerceIn(0f, height.toFloat())
        if (l >= r || t >= b) return emptyList()

        val c0 = (l / tilesize).toInt().coerceIn(0, tilesCols - 1)
        val c1 = ((r - 1f) / tilesize).toInt().coerceIn(0, tilesCols - 1)
        val row0 = (t / tilesize).toInt().coerceIn(0, tilesRows - 1)
        val row1 = ((b - 1f) / tilesize).toInt().coerceIn(0, tilesRows - 1)

        val result = ArrayList<TileRect>((c1 - c0 + 1) * (row1 - row0 + 1))
        for (row in row0..row1) {
            for (col in c0..c1) {
                val idx = row * tilesCols + col
                result.add(
                    TileRect(
                        textures[idx],
                        textureViews[idx],
                        col * tilesize,
                        row * tilesize,
                        tileUniformFor(idx)
                    )
                )
            }
        }
        return result
    }

    // Cache the last computed quad to avoid allocations when panning within the same tile region
    private var lastQuadTX = -1
    private var lastQuadTY = -1
    private var lastQuad: Quad? = null

    fun getQuad(centerX: Int, centerY: Int): Quad {
        cachedQuad?.let { return it }

        val cX = centerX.toFloat()
        val cY = centerY.toFloat()

        val c = (cX / tilesize).toInt()
        val tX = when {
            c >= tilesCols - 1 -> tilesCols - 2
            c <= 0 -> 0
            else -> {
                val xCenterRight = if (c + 1 == tilesCols - 1) {
                    ((tilesCols - 1) * tilesize + width) * 0.5
                } else {
                    (c + 1.5) * tilesize
                }

                if (cX - (c - 0.5) * tilesize < xCenterRight - cX) c - 1 else c
            }
        }.coerceIn(0, tilesCols - 1)

        val r = (cY / tilesize).toInt()
        val tY = when {
            r >= tilesRows - 1 -> tilesRows - 2
            r <= 0 -> 0
            else -> {
                val yCenterBottom = if (r + 1 == tilesRows - 1) {
                    ((tilesRows - 1) * tilesize + height) * 0.5
                } else {
                    (r + 1.5) * tilesize
                }

                if (cY - (r - 0.5) * tilesize < yCenterBottom - cY) r - 1 else r
            }
        }.coerceIn(0, tilesRows - 1)

        // Return cached quad if tile region hasn't changed
        lastQuad?.let { cached ->
            if (lastQuadTX == tX && lastQuadTY == tY) {
                return cached
            }
        }

        // Build a 2x2 quad from the tile grid at (tX, tY) without mutableList overhead
        val r0 = (tY).coerceAtMost(tilesRows - 1) * tilesCols
        val r1 = (tY + 1).coerceAtMost(tilesRows - 1) * tilesCols
        val c0 = (tX).coerceAtMost(tilesCols - 1)
        val c1 = (tX + 1).coerceAtMost(tilesCols - 1)

        val t00 = textures[r0 + c0]
        val v00 = textureViews[r0 + c0]
        val t01 = textures[r0 + c1]
        val v01 = textureViews[r0 + c1]
        val t10 = textures[r1 + c0]
        val v10 = textureViews[r1 + c0]
        val t11 = textures[r1 + c1]
        val v11 = textureViews[r1 + c1]

        val quad = Quad(
            listOf(t00, t01, t10, t11),
            listOf(v00, v01, v10, v11),
            tX * tilesize,
            tY * tilesize
        )
        lastQuadTX = tX
        lastQuadTY = tY
        lastQuad = quad
        return quad
    }
}
