package ca.mpreg.webgpuviewer

import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
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
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.round

class Image private constructor(val width: Int, val height: Int) {
    companion object {
        val device get() = WebGpuRenderer.device
        val dispatcher get() = WebGpuRenderer.dispatcher
        var pipeline: GPURenderPipeline

        init {
            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(ImageShader))
            )

            pipeline = device.createRenderPipeline(
                GPURenderPipelineDescriptor(
                    vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                    fragment = GPUFragmentState(
                        shaderModule,
                        entryPoint = "fs_main",
                        targets = arrayOf(GPUColorTargetState(format = TextureFormat.RGBA8Unorm))
                    ),
                    primitive = GPUPrimitiveState(topology = TriangleList),
                )
            )
        }

        suspend operator fun invoke(pixels: ByteBuffer, width: Int, height: Int): Image {
            return Image(width, height).apply {
                val tilesize = 4096
                val maxWidth = 4096
                val maxHeight = 4096

                var pixels = pixels

                withContext(dispatcher) {
                    buffer = device.createBuffer(
                        GPUBufferDescriptor(
                            size = 32, usage = BufferUsage.CopyDst or BufferUsage.Uniform
                        )
                    )

                    mipmaps.add(Mipmap(pixels, width, height, 1f, tilesize))

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

                        render(encoder, texture, 0f, 0f, scale)
                        device.queue.submit(arrayOf(encoder.finish()))

                        mipmaps.add(Mipmap(texture, scale, tilesize))
                    }

                    Log.i("Renderer", "Finished create mipmaps")
                }
            }
        }
    }

    var x: Float = 0f
    var y: Float = 0f

    val mipmaps: MutableList<Mipmap> = mutableListOf()

    val byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(32).apply {
        order(ByteOrder.nativeOrder())
    }

    var buffer: GPUBuffer? = null

    fun update(pixels: ByteBuffer) {
        mipmaps[0].update(pixels)
    }

    protected fun finalize() {
        buffer?.close()
    }

    fun render(encoder: GPUCommandEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float) {
        if (mipmaps.isEmpty()) return
        val buffer = buffer ?: return

        val level = floor(log2(1 / scale)).toInt().coerceIn(0, mipmaps.size - 1)

        val x = x + this.x / dst.width
        val y = y + this.y / dst.height

        val mipmap = mipmaps[level]

        val vx = round(-x * dst.width + mipmap.width / 2).toInt()
        val vy = round(-y * dst.height + mipmap.height / 2).toInt()

        val quad = mipmap.getQuad(vx, vy)

        byteBuffer.putFloat(
            0, (0.5f / scale + x) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width
        )
        byteBuffer.putFloat(
            4, (0.5f / scale + y) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height
        )
        byteBuffer.putFloat(8, scale / mipmap.scale)
        byteBuffer.putFloat(12, mipmap.tilesize.toFloat())
        byteBuffer.putFloat(16, mipmap.tilesCols.toFloat())
        byteBuffer.putFloat(20, mipmap.tilesRows.toFloat())
        byteBuffer.putFloat(24, dst.width.toFloat())
        byteBuffer.putFloat(28, dst.height.toFloat())
        device.queue.writeBuffer(buffer, 0, byteBuffer)

        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Load,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 1.0)
                    )
                )
            )
        )

        pass.setPipeline(pipeline)
        pass.setBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                        GPUBindGroupEntry(0, buffer = buffer),
                    ).plus(quad.tiles.mapIndexed { i, value ->
                        GPUBindGroupEntry(1 + i, textureView = value.createView())
                    })
                )
            )
        )

        pass.draw(6)
        pass.end()
    }
}
