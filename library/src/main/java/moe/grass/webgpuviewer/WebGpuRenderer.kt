package moe.grass.webgpuviewer

import android.graphics.Bitmap
import android.util.Log
import android.view.Surface
import androidx.webgpu.BufferUsage
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostException
import androidx.webgpu.FeatureLevel
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPUInstanceDescriptor
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUSurface
import androidx.webgpu.GPUSurfaceConfiguration
import androidx.webgpu.GPUSurfaceDescriptor
import androidx.webgpu.GPUSurfaceSourceAndroidNativeWindow
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.createGpuTexture
import androidx.webgpu.helper.initLibrary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executor
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

object webgpu {
    var instance: GPUInstance? = null
    var adapter: GPUAdapter? = null
    var device: GPUDevice? = null
    var pipeline: GPURenderPipeline? = null
}

class WebGpuRenderer {
    private lateinit var surface: GPUSurface

    var tilesize = 4096

    var width: Int = 0
    var height: Int = 0

    var image_width: Int = 0
    var image_height: Int = 0

    var min_scale: Float = 0f

    var ready = false

    class Mipmap(
        val scale: Float,
        val tiles: List<List<GPUTexture>>,
        val width: Int,
        val height: Int
    ) {
        constructor(scale: Float, tiles: List<List<GPUTexture>>) : this(
            scale,
            tiles,
            tiles[0].size,
            tiles.size
        )
    }

    var mipmaps: MutableList<Mipmap> = mutableListOf()

    var byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(32)
    private lateinit var buffer: GPUBuffer

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    suspend fun createTiles(
        device: GPUDevice,
        image: Bitmap
    ): MutableList<MutableList<GPUTexture>> {
        Log.i("webgpuviewer", "Create tiles " + image.width + " " + image.height)
        val tiles: MutableList<MutableList<GPUTexture>> = mutableListOf()
        for (y in 0 until image.height step tilesize) {
            val col: MutableList<GPUTexture> = mutableListOf()
            tiles.add(col)
            for (x in 0 until image.width step tilesize) {
                Log.i("webgpuviewer", "Create tile " + x + " " + y)
                val width = min(tilesize, image.width - x)
                val height = min(tilesize, image.height - y)
                val cropped = withContext(Dispatchers.Default) {
                    Bitmap.createBitmap(image, x, y, width, height)
                }
                val texture = cropped.createGpuTexture(device)
                col.add(texture)
            }
        }
        return tiles
    }

    suspend fun init(image: Bitmap, surface: Surface, width: Int, height: Int) {
        initLibrary()

        if (webgpu.instance == null) {
            webgpu.instance = createInstance(GPUInstanceDescriptor())
            webgpu.adapter =
                webgpu.instance!!.requestAdapter(GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility))
            webgpu.device = webgpu.adapter!!.requestDevice(
                GPUDeviceDescriptor(
                    deviceLostCallback = defaultDeviceLostCallback,
                    deviceLostCallbackExecutor = Executor(Runnable::run),
                    uncapturedErrorCallback = defaultUncapturedErrorCallback,
                    uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                )
            )

            val shaderModule = webgpu.device!!.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(
                        WebGpuRendererShader.shader
                    )
                )
            )

            webgpu.pipeline = webgpu.device!!.createRenderPipeline(
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

        this.surface = surface.let {
            webgpu.instance!!.createSurface(
                GPUSurfaceDescriptor(
                    surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                        windowFromSurface(it)
                    )
                )
            )
        }

        this.surface.configure(
            GPUSurfaceConfiguration(
                webgpu.device!!,
                width,
                height,
                TextureFormat.RGBA8Unorm,
                TextureUsage.RenderAttachment
            )
        )

        val device = webgpu.device!!

        this.width = width
        this.height = height

        val ratiox = width.toFloat() / image.width.toFloat()
        val ratioy = height.toFloat() / image.height.toFloat()

        if (this.image_width == 0) {
            this.x = 0f
            this.y = 0f
            this.min_scale = max(0.01f, min(ratiox, ratioy))
            this.scale = this.min_scale
        }

        this.image_width = image.width
        this.image_height = image.height
        byteBuffer.order(ByteOrder.nativeOrder())

        buffer = device.createBuffer(
            GPUBufferDescriptor(
                size = 32,
                usage = BufferUsage.CopyDst or BufferUsage.Uniform
            )
        )

        mipmaps.forEach { it.tiles.flatten().forEach { it.destroy() } }
        mipmaps.clear()
        mipmaps.add(Mipmap(1f, createTiles(device, image)))

        Log.i("webgpuviewer", "Create mipmaps")
        var scale = 1f
        while (image_width * scale > 4096 && image_height * scale > 4096) {
            scale /= 2
            val im = withContext(Dispatchers.Default) {
                ImageUtil.resize(
                    image,
                    (image_width * scale).toInt(),
                    (image_height * scale).toInt()
                )
            }
            mipmaps.add(Mipmap(scale, createTiles(device, im)))
        }

        while (image_width * scale > 512 && image_height * scale > 512) {
            scale /= 2
            val texture = device.createTexture(
                GPUTextureDescriptor(
                    size = GPUExtent3D(
                        (image_width * scale).toInt(),
                        (image_height * scale).toInt(),
                        1
                    ),
                    usage = TextureUsage.TextureBinding or TextureUsage.RenderAttachment,
                    format = TextureFormat.RGBA8Unorm
                )
            )
            render(mipmaps[mipmaps.size - 1], texture, 0f, 0f, scale)
            mipmaps.add(Mipmap(scale, listOf(listOf(texture))))
        }

        ready = true
    }

    fun render() {
        if (mipmaps.size == 0) {
            return
        }

        var level = floor(log2(1 / scale)).toInt()
        level = max(min(level, mipmaps.size - 1), 0)
        render(mipmaps[level], surface.getCurrentTexture().texture, x, y, scale)
        surface.present()
    }

    fun render(mipmap: Mipmap, dst: GPUTexture, x: Float, y: Float, scale: Float) {
        val commandEncoder = webgpu.device!!.createCommandEncoder()

        var vx = round((-x * dst.width + 0.5 * image_width) * mipmap.scale / tilesize).toInt() - 1
        vx = min(vx, mipmap.width - 2)
        vx = max(vx, 0)
        var vy = round((-y * dst.height + 0.5 * image_height) * mipmap.scale / tilesize).toInt() - 1
        vy = min(vy, mipmap.height - 2)
        vy = max(vy, 0)

        byteBuffer.putFloat(
            0,
            (0.5f / scale + x) * mipmap.scale + (vx * tilesize - (mipmap.scale * image_width) / 2f) / dst.width
        )
        byteBuffer.putFloat(
            4,
            (0.5f / scale + y) * mipmap.scale + (vy * tilesize - (mipmap.scale * image_height) / 2f) / dst.height
        )
        byteBuffer.putFloat(8, scale / mipmap.scale)
        byteBuffer.putFloat(12, tilesize.toFloat())
        byteBuffer.putFloat(16, mipmap.width.toFloat())
        byteBuffer.putFloat(20, mipmap.height.toFloat())
        byteBuffer.putFloat(24, dst.width.toFloat())
        byteBuffer.putFloat(28, dst.height.toFloat())
        webgpu.device!!.queue.writeBuffer(buffer, 0, byteBuffer)

        val vx1 = if (mipmap.width > 1) vx + 1 else vx
        val vy1 = if (mipmap.height > 1) vy + 1 else vy

        val textures = arrayOf(
            mipmap.tiles[vy][vx],
            mipmap.tiles[vy][vx1],
            mipmap.tiles[vy1][vx],
            mipmap.tiles[vy1][vx1],
        ).mapIndexed { index, texture ->
            GPUBindGroupEntry(binding = 1 + index, textureView = texture.createView())
        }

        val pass = commandEncoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(), loadOp = LoadOp.Clear, storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 1.0)
                    )
                )
            )
        )

        pass.setPipeline(webgpu.pipeline!!)
        pass.setBindGroup(
            0, webgpu.device!!.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = webgpu.pipeline!!.getBindGroupLayout(0),
                    entries = arrayOf(
                        GPUBindGroupEntry(
                            binding = 0,
                            buffer = buffer
                        )
                    ).plus(textures)
                )
            )
        )
        pass.draw(3)
        pass.end()
        webgpu.device!!.queue.submit(arrayOf(commandEncoder.finish()))
    }

    fun cleanup() {
        surface.close()
        buffer.close()
        mipmaps.forEach { it.tiles.flatten().forEach { it.destroy() } }
        mipmaps.clear()
    }
}

private val defaultUncapturedErrorCallback
    get(): UncapturedErrorCallback {
        return UncapturedErrorCallback { _, type, message ->
            throw WebGpuRuntimeException.create(type, message)
        }
    }

private val defaultDeviceLostCallback
    get(): DeviceLostCallback {
        return DeviceLostCallback { device, reason, message ->
            throw DeviceLostException(device, reason, message)
        }
    }
