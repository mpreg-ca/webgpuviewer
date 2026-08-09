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

                // Prepare all mipmap pixel data on CPU first (no GPU mutex needed)
                data class MipmapData(val pixels: ByteBuffer, val w: Int, val h: Int, val scale: Float)
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

                // Now upload all textures to GPU in one batch (minimizes mutex holding time)
                WebGpuRenderer.withContext { device ->
                    try {
                        for (data in mipmapDataList) {
                            mipmaps.add(Mipmap(data.pixels, data.w, data.h, data.scale, tilesize))
                        }

                        // GPU-based mipmaps if needed
                        if (createMipMaps && mipmapDataList.isNotEmpty()) {
                            var scale = mipmapDataList.last().scale
                            while (width * scale > maxWidth && height * scale > maxHeight) {
                                scale /= 2
                                val newWidth = floor(width * scale).toInt()
                                val newHeight = floor(height * scale).toInt()
                                Log.d("Renderer", "Create mipmap using shader ${scale} ${newWidth} ${newHeight}")
                                
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
                        }
                    } catch (e: Exception) {
                        Log.e("Renderer", "Error creating mipmaps", e)
                        // Clean up any mipmaps that were created before the error
                        mipmaps.forEach { it.cleanup() }
                        mipmaps.clear()
                        throw e
                    }
                }
            }
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
        _buffer?.close()
        _buffer = null
    }

    class MipMapForDraw(
        val mipmap: Mipmap, val quad: Mipmap.Quad, val x: Float, val y: Float, val scale: Float
    )

    fun prepareForRender(dst: GPUTexture, x: Float, y: Float, scale: Float): MipMapForDraw? {
        if (mipmaps.isEmpty()) return null

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        val mipmap = mipmaps[level]

        val x = x + this.x / dst.width + WebGpuRenderer.offsetX
        val y = y + this.y / dst.height + WebGpuRenderer.offsetY

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
