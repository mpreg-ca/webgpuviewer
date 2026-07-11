package ca.mpreg.webgpuviewer.renderer

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
    companion object {
        suspend operator fun invoke(
            pixels: ByteBuffer, width: Int, height: Int, createMipMaps: Boolean = true
        ): Image {
            return Image(width, height).apply {
                val tilesize = 4096
                val maxWidth = 4096
                val maxHeight = 4096

                var pixels = pixels

                WebGpuRenderer.withContext { device ->
                    mipmaps.add(Mipmap(pixels, width, height, 1f, tilesize))

                    if (!createMipMaps) return@withContext

                    var scale = 1f

                    Log.i("Renderer", "Creating mipmaps")

                    var textureWidth = width
                    var textureHeight = height

                    while (width * scale > tilesize || height * scale > tilesize) {
                        scale /= 2
                        val newWidth = floor(width * scale).toInt()
                        val newHeight = floor(height * scale).toInt()
                        Log.i(
                            "Renderer", "Create mipmap using CPU ${scale} ${newWidth} ${newHeight}"
                        )
                        // TODO: return mutex
                        pixels = withContext(Dispatchers.Default) {
                            ImageUtil.resize(pixels, textureWidth, textureHeight)
                        }

                        mipmaps.add(Mipmap(pixels, newWidth, newHeight, scale, tilesize))

                        textureWidth = newWidth
                        textureHeight = newHeight
                    }

                    while (width * scale > maxWidth && height * scale > maxHeight) {
                        scale /= 2
                        val newWidth = floor(width * scale).toInt()
                        val newHeight = floor(height * scale).toInt()
                        Log.i(
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

                        TransitionBasic.render(this@apply, encoder, texture, 0f, 0f, scale)
                        device.queue.submit(arrayOf(encoder.finish()))

                        mipmaps.add(Mipmap(texture, scale, tilesize))
                    }

                    Log.i("Renderer", "Finished create mipmaps")
                }
            }
        }
    }

    val buffer: GPUBuffer by lazy {
        WebGpuRenderer.device.createBuffer(
            GPUBufferDescriptor(
                size = BUFFER_SIZE, usage = BufferUsage.CopyDst or BufferUsage.Uniform
            )
        )
    }

    val mipmaps: MutableList<Mipmap> = mutableListOf()

    fun update(pixels: ByteBuffer) {
        mipmaps[0].update(pixels)
    }

    protected fun finalize() {
        buffer.close()
    }

    class MipMapForDraw(
        val mipmap: Mipmap, val quad: Mipmap.Quad, val x: Float, val y: Float, val scale: Float
    )

    fun prepareForRender(dst: GPUTexture, x: Float, y: Float, scale: Float): MipMapForDraw? {
        if (mipmaps.isEmpty()) return null

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        val mipmap = mipmaps[level]

        val x = x + this.x / dst.width
        val y = y + this.y / dst.height

        val vx = round(-x * dst.width + mipmap.width / 2).toInt()
        val vy = round(-y * dst.height + mipmap.height / 2).toInt()

        val quad = mipmap.getQuad(vx, vy)

        return MipMapForDraw(
            mipmap,
            quad,
            (0.5f / scale + x) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width,
            (0.5f / scale + y) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height,
            scale / mipmap.scale
        )
    }
}
