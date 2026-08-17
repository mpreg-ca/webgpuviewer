package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.ImageUtil
import ca.mpreg.webgpuviewer.Trim
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.round

const val BUFFER_SIZE = 96L

class Image private constructor(
    val width: Int, val height: Int, var x: Float = 0f, var y: Float = 0f
) {
    /**
     * Background color detected from image edges, in 0xAARRGGBB format.
     * Defaults to opaque black (0xFF000000).
     */
    var backgroundColor: Int = 0xFF000000.toInt()

    /**
     * Trim bounds detected from image content, or null if not trimmed.
     */
    var trim: Rect? = null

    /**
     * Position of this image in a spread.
     */
    enum class Position {
        LEFT,
        RIGHT,
        SINGLE
    }

    var position: Position = Position.SINGLE

    companion object {
        /**
         * When true, mipmap levels below the CPU-generated ones are produced on the GPU with a
         * render pass. Disabled by default: all mipmap levels are generated on the CPU.
         */
        var useGpuMipmaps: Boolean = false

        suspend fun createWithTrim(
            pixels: ByteBuffer, width: Int, height: Int,
            createMipMaps: Boolean = true,
            trimColors: List<FloatArray>? = null,
            trimThreshold: Float = 0.05f,
            backgroundColor: Int? = null,
        ): Image {
            require(width > 0 && height > 0) { "Image dimensions must be positive" }
            require(trimColors == null || trimColors.all { it.size >= 3 }) {
                "each trimColor must have at least 3 elements [r, g, b]"
            }

            val image = Image(width, height)

            // Trim and background detection read the decoded pixels, so they run on a background
            // dispatcher here instead of as compute shaders. The GPU versions have to park on a
            // buffer readback while holding the render thread, which stalls every frame queued
            // behind them - visible as a stutter each time a page decodes.
            withContext(Dispatchers.Default) {
                var backgroundFromTrim = false

                val trimWith = trimColors?.takeIf { it.isNotEmpty() }
                if (trimWith != null) {
                    // Find trim for each color and pick the smallest rect
                    val rects = Trim.findAllCpu(pixels, width, height, trimWith, trimThreshold)
                    val best = trimWith.zip(rects)
                        .minByOrNull { it.second.width() * it.second.height() }

                    if (best != null) {
                        image.trim = best.second
                        // Set background color from the winning trim color
                        if (backgroundColor == null) {
                            val c = best.first
                            image.backgroundColor = 0xFF000000.toInt() or
                                    ((c[0] * 255).toInt() shl 16) or
                                    ((c[1] * 255).toInt() shl 8) or (c[2] * 255).toInt()
                            backgroundFromTrim = true
                        }
                    }
                }

                // Probing the edges is only worth a pass when neither the caller nor trim has
                // already named a background colour.
                if (backgroundColor != null) {
                    image.backgroundColor = backgroundColor
                } else if (!backgroundFromTrim) {
                    image.backgroundColor =
                        Trim.detectBackgroundCpu(pixels, width, height, trimThreshold)
                }
            }

            val tilesize = 2048
            val maxWidth = 4096
            val maxHeight = 4096

            data class MipmapData(
                val pixels: ByteBuffer, val w: Int, val h: Int, val scale: Float
            )

            val mipmapDataList = mutableListOf<MipmapData>()
            mipmapDataList.add(MipmapData(pixels, width, height, 1f))

            if (createMipMaps) {
                var currentPixels = pixels
                var textureWidth = width
                var textureHeight = height
                var scale = 1f

                // With the GPU path enabled the CPU only has to reach the tile size; the shader
                // takes the remaining levels down to maxWidth/maxHeight. Otherwise the CPU has to
                // cover those levels too.
                val cpuMaxWidth = if (useGpuMipmaps) tilesize else minOf(tilesize, maxWidth)
                val cpuMaxHeight = if (useGpuMipmaps) tilesize else minOf(tilesize, maxHeight)

                while (width * scale > cpuMaxWidth || height * scale > cpuMaxHeight) {
                    scale /= 2
                    val newWidth = floor(width * scale).toInt()
                    val newHeight = floor(height * scale).toInt()
                    Log.d("Renderer", "Create mipmap using CPU ${scale} ${newWidth} ${newHeight}")

                    currentPixels = withContext(Dispatchers.Default) {
                        ImageUtil.resize(currentPixels, textureWidth, textureHeight)
                    }
                    mipmapDataList.add(MipmapData(currentPixels, newWidth, newHeight, scale))
                    textureWidth = newWidth
                    textureHeight = newHeight
                }
            }

            // No render mutex: Mipmap.create yields between upload chunks so queued frames get
            // the GPU thread back, and holding the mutex would make those yields pointless. Safe
            // because the image isn't reachable from any page until this function returns, so
            // nothing can sample or destroy it while the upload is in flight.
            WebGpuRenderer.onDispatcher { device ->
                try {
                    for (data in mipmapDataList) {
                        image.mipmaps.add(
                            Mipmap.create(data.pixels, data.w, data.h, data.scale, tilesize)
                        )
                    }

                    if (useGpuMipmaps && createMipMaps && mipmapDataList.isNotEmpty()) {
                        var scale = mipmapDataList.last().scale
                        while (width * scale > maxWidth || height * scale > maxHeight) {
                            scale /= 2
                            val newWidth = floor(width * scale).toInt()
                            val newHeight = floor(height * scale).toInt()
                            Log.d(
                                "Renderer",
                                "Create mipmap using shader ${scale} ${newWidth} ${newHeight}"
                            )

                            val size = GPUExtent3D(newWidth, newHeight)
                            val texture = device.createTexture(
                                GPUTextureDescriptor(
                                    size = size,
                                    usage = TextureUsage.TextureBinding or TextureUsage.RenderAttachment,
                                    format = TextureFormat.RGBA8Unorm
                                )
                            )
                            val encoder = device.createCommandEncoder()
                            // Fresh texture, so clear rather than load - there is nothing to
                            // preserve and loading would cost a pointless tile read.
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
                                RenderPage.render(pass, image, texture, 0f, 0f, scale)
                            } finally {
                                pass.end()
                            }
                            device.queue.submit(arrayOf(encoder.finish()))
                            image.mipmaps.add(Mipmap(texture, scale, tilesize))
                        }
                    }
                } catch (e: Exception) {
                    Log.e("Renderer", "Error creating image", e)
                    image.mipmaps.forEach { it.cleanup() }
                    image.mipmaps.clear()
                    throw e
                }
            }

            return image
        }

        suspend operator fun invoke(
            pixels: ByteBuffer, width: Int, height: Int, createMipMaps: Boolean = true
        ): Image {
            return createWithTrim(pixels, width, height, createMipMaps, null)
        }

        suspend operator fun invoke(width: Int, height: Int): Image {
            return Image(width, height).apply {
                WebGpuRenderer.withContext { _ ->
                    try {
                        mipmaps.add(Mipmap(width, height))
                    } catch (e: Exception) {
                        Log.e("Renderer", "Error creating drawable image", e)
                        throw e
                    }
                }
            }
        }
    }

    private var _buffer: GPUBuffer? = WebGpuRenderer.device.createBuffer(
        GPUBufferDescriptor(size = BUFFER_SIZE, usage = BufferUsage.CopyDst or BufferUsage.Uniform)
    )

    val buffer: GPUBuffer
        get() = _buffer ?: error("Image buffer accessed after cleanup")

    val mipmaps: MutableList<Mipmap> = mutableListOf()

    internal fun cleanup() {
        mipmaps.forEach { it.cleanup() }
        mipmaps.clear()
        _buffer?.destroy()
        _buffer = null
    }

    class MipMapForDraw(
        val mipmap: Mipmap, val quad: Mipmap.Quad, val x: Float, val y: Float, val scale: Float
    )

    fun prepareForRender(dst: GPUTexture, x: Float, y: Float, scale: Float): MipMapForDraw? {
        if (mipmaps.isEmpty()) return null

        var level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        // Scale alone isn't enough to pick a level. A draw binds a 2x2 window of tiles, and
        // getQuad can only promise half a tile either side of the view centre, so the visible
        // span has to fit within one tile's worth of source texels. A level whose whole grid is
        // 2x2 or smaller is bound in one go and always fits; anything larger has to be checked.
        //
        // Without this, a page taller than two tiles renders only the window around the centre
        // and the rest of it comes out empty. Stepping coarser costs nothing in sharpness here:
        // the level that fits is the one roughly matching the size it's drawn at.
        while (level < mipmaps.size - 1) {
            val m = mipmaps[level]
            if (m.tilesCols <= 2 && m.tilesRows <= 2) break

            // Source texels the viewport covers at this level.
            val visibleW = dst.width * m.scale / scale
            val visibleH = dst.height * m.scale / scale
            if (visibleW <= m.tilesize && visibleH <= m.tilesize) break

            level++
        }

        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        // View centre in this level's pixels: -adjustedX * dst.width is the offset from the
        // image centre in level-0 pixels, so it scales by mipmap.scale before adding the level's
        // half-size. Without the factor the quad window lands up to 2^level too far out - masked
        // on screen because the guard above almost always ends on a <= 2x2 grid (where there is
        // only one window), but the tile cache renders 256px targets that legitimately pick fine
        // levels with large grids.
        val vx = round(-adjustedX * dst.width * mipmap.scale + mipmap.width / 2).toInt()
        val vy = round(-adjustedY * dst.height * mipmap.scale + mipmap.height / 2).toInt()

        val quad = mipmap.getQuad(vx, vy)

        return MipMapForDraw(
            mipmap,
            quad,
            (0.5f / scale + adjustedX) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width,
            (0.5f / scale + adjustedY) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height,
            scale / mipmap.scale
        )
    }
}
