package ca.mpreg.webgpuviewer

import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.round

class Image(val width: Int, val height: Int) {
    var x: Float = 0f
    var y: Float = 0f

    var mipmaps: MutableList<Mipmap> = mutableListOf()

    var byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(32)
    lateinit var buffer: GPUBuffer

    constructor(pixels: ByteBuffer, width: Int, height: Int) : this(width, height) {
        byteBuffer.order(ByteOrder.nativeOrder())

        buffer = webgpu.device.createBuffer(
            GPUBufferDescriptor(
                size = 32,
                usage = BufferUsage.CopyDst or BufferUsage.Uniform
            )
        )

        val tilesize = 4096
        val maxWidth = 1024
        val maxHeight = 1024

        mipmaps.add(Mipmap(pixels, width, height, 1f, tilesize))

        var scale = 1f

        CoroutineScope(webgpu.dispatcher).launch {
            Log.i("Renderer", "Creating mipmaps")

            while (width * scale > tilesize || height * scale > tilesize) {
                scale /= 2
                Log.i(
                    "Renderer",
                    "Create mipmap using CPU ${scale} ${width * scale} ${height * scale}"
                )
                val im = withContext(Dispatchers.Default) {
                    ImageUtil.resize(
                        pixels,
                        (width * scale).toInt(),
                        (height * scale).toInt()
                    )
                }
                mipmaps.add(
                    Mipmap(
                        im,
                        (width * scale).toInt(),
                        (height * scale).toInt(),
                        scale,
                        tilesize
                    )
                )
            }

            while (width * scale > maxWidth && height * scale > maxHeight) {
                scale /= 2
                Log.i(
                    "Renderer",
                    "Create mipmap using shader ${scale} ${width * scale} ${height * scale}"
                )
                val size = GPUExtent3D((width * scale).toInt(), (height * scale).toInt())
                val texture = webgpu.device.createTexture(
                    GPUTextureDescriptor(
                        size = size,
                        usage = TextureUsage.TextureBinding or TextureUsage.RenderAttachment,
                        format = TextureFormat.RGBA8Unorm
                    )
                )
                val encoder = webgpu.device.createCommandEncoder()

                render(encoder, texture, 0f, 0f, scale)
                webgpu.device.queue.submit(arrayOf(encoder.finish()))

                mipmaps.add(Mipmap(texture, scale, tilesize))
            }

            Log.i("Renderer", "Finished create mipmaps")
        }
    }

    fun update(pixels: ByteBuffer) {
        mipmaps[0].update(pixels)
    }

    fun cleanup() {
        mipmaps.forEach { it.cleanup() }
        buffer.close()
    }

    fun render(encoder: GPUCommandEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float) {
        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        val x = x + this.x / dst.width
        val y = y + this.y / dst.height

        val mipmap = mipmaps[level]

        val vx = round(((-x * width / mipmap.width + 0.5) * mipmap.width)).toInt()
        val vy = round(((-y * height / mipmap.height + 0.5) * mipmap.height)).toInt()

        val quad = mipmap.getQuad(vx, vy)

        byteBuffer.putFloat(
            0,
            (0.5f / scale + x) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width
        )
        byteBuffer.putFloat(
            4,
            (0.5f / scale + y) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height
        )
        byteBuffer.putFloat(8, scale / mipmap.scale)
        byteBuffer.putFloat(12, mipmap.tilesize.toFloat())
        byteBuffer.putFloat(16, mipmap.tilesCols.toFloat())
        byteBuffer.putFloat(20, mipmap.tilesRows.toFloat())
        byteBuffer.putFloat(24, dst.width.toFloat())
        byteBuffer.putFloat(28, dst.height.toFloat())
        webgpu.device.queue.writeBuffer(buffer, 0, byteBuffer)

        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(), loadOp = LoadOp.Load, storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 1.0)
                    )
                )
            )
        )

        pass.setPipeline(webgpu.pipeline)
        pass.setBindGroup(
            0, webgpu.device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = webgpu.pipeline.getBindGroupLayout(0),
                    entries = arrayOf(
                        GPUBindGroupEntry(
                            binding = 0,
                            buffer = buffer
                        ),
                    ).plus(quad.tiles.mapIndexed { i, value ->
                        GPUBindGroupEntry(
                            binding = 1 + i,
                            textureView = value.createView()
                        )
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }
}
