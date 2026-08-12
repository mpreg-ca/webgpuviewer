package ca.mpreg.webgpuviewer.renderer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import ca.mpreg.webgpuviewer.ImageUtil
import ca.mpreg.webgpuviewer.Trim
import ca.mpreg.webgpuviewer.transition.TransitionBasic
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
            var trimRect: Rect? = null

            val tilesize = 4096
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

                while (width * scale > tilesize || height * scale > tilesize) {
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

            WebGpuRenderer.withContext { device ->
                try {
                    for (data in mipmapDataList) {
                        image.mipmaps.add(Mipmap(data.pixels, data.w, data.h, data.scale, tilesize))
                    }

                    if (createMipMaps && mipmapDataList.isNotEmpty()) {
                        var scale = mipmapDataList.last().scale
                        while (width * scale > maxWidth && height * scale > maxHeight) {
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
                            TransitionBasic.render(image, encoder, texture, 0f, 0f, scale)
                            device.queue.submit(arrayOf(encoder.finish()))
                            image.mipmaps.add(Mipmap(texture, scale, tilesize))
                        }
                    }

                    if (!trimColors.isNullOrEmpty() && image.mipmaps.isNotEmpty()) {
                        // Find trim for each color and pick the smallest rect
                        val results = trimColors.map { color ->
                            color to Trim.findInContext(
                                image, color[0], color[1], color[2], trimThreshold
                            )
                        }

                        val best = results.minByOrNull { it.second.width() * it.second.height() }
                        if (best != null) {
                            trimRect = best.second
                            image.trim = trimRect  // Store trim on image
                            // Set background color from the winning trim color
                            if (backgroundColor == null) {
                                val c = best.first
                                image.backgroundColor = 0xFF000000.toInt() or
                                        ((c[0] * 255).toInt() shl 16) or ((c[1] * 255).toInt() shl 8) or (c[2] * 255).toInt()
                            }
                        }
                    }

                    // Use explicitly provided background color if given, otherwise detect
                    image.backgroundColor =
                        backgroundColor ?: Trim.detectBackgroundInContext(image)
                } catch (e: Exception) {
                    Log.e("Renderer", "Error creating image with trim", e)
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

    private var _buffer: GPUBuffer? = null

    val buffer: GPUBuffer
        get() {
            if (_buffer == null) {
                _buffer = WebGpuRenderer.device.createBuffer(
                    GPUBufferDescriptor(
                        size = BUFFER_SIZE, usage = BufferUsage.CopyDst or BufferUsage.Uniform
                    )
                )
            }
            return _buffer!!
        }

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

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        val mipmap = mipmaps[level]

        val adjustedX = x + this.x / dst.width + WebGpuRenderer.offsetX
        val adjustedY = y + this.y / dst.height + WebGpuRenderer.offsetY

        val vx = round(-adjustedX * dst.width + mipmap.width / 2).toInt()
        val vy = round(-adjustedY * dst.height + mipmap.height / 2).toInt()

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
