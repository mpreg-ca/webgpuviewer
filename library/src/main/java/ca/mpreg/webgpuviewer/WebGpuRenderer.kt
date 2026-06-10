package ca.mpreg.webgpuviewer

import android.util.Log
import android.view.Surface
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.tween
import androidx.compose.ui.geometry.Offset
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostException
import androidx.webgpu.FeatureLevel
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
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
import androidx.webgpu.GPUVertexState
import androidx.webgpu.LoadOp
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.initLibrary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import kotlin.math.abs
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
    private var surface: GPUSurface? = null

    var width: Int = 0
    var height: Int = 0

    val imageWidth: Int
        get() = images.map { it.width - it.position.x }.max() - images.map { it.position.x }.min()
    val imageHeight: Int
        get() = images.map { it.height - it.position.y }.max() - images.map { it.position.y }.min()

    var animationJob: Job? = null

    var images: MutableList<Image> = mutableListOf()

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var minScale = 0.5f
    var maxScale = 2f

    suspend fun init(
        surface: Surface,
        width: Int,
        height: Int,
    ) {
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

        this.width = width
        this.height = height

        this.surface = surface.let {
            webgpu.instance!!.createSurface(
                GPUSurfaceDescriptor(
                    surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                        windowFromSurface(it)
                    )
                )
            )
        }

        this.surface!!.configure(
            GPUSurfaceConfiguration(
                webgpu.device!!,
                width,
                height,
                TextureFormat.RGBA8Unorm,
                TextureUsage.RenderAttachment
            )
        )
    }

    fun maxX(scale: Float = this.scale): Float {
        return max(0f, (imageWidth.toFloat() / width - 1 / scale) / 2)
    }

    fun maxY(scale: Float = this.scale): Float {
        return max(0f, (imageHeight.toFloat() / height - 1 / scale) / 2)
    }

    fun setPos(x: Float, y: Float) {
        if (this.x == x && this.y == y) {
            return
        }
        this.x = x
        this.y = y
        render()
    }

    fun render() {
        if (surface == null) {
            return
        }

        CoroutineScope(Dispatchers.Main).launch {
            val texture = surface!!.getCurrentTexture().texture
            val commandEncoder = webgpu.device!!.createCommandEncoder()

            val clearPass = commandEncoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = texture.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 1.0)
                        )
                    )
                )
            )
            clearPass.end()

            images.forEach {
                var level = floor(log2(1 / scale)).toInt()
                level = max(min(level, it.mipmaps.size - 1), 0)
                Log.i(
                    "webgpurenderer",
                    "render at ${it.position.x} ${width} ${it.position.x.toFloat() / width}"
                )
                render(
                    commandEncoder,
                    it,
                    level,
                    texture,
                    x + it.position.x.toFloat() / width,
                    y + it.position.y.toFloat() / height,
                    scale
                )
            }
            webgpu.device!!.queue.submit(arrayOf(commandEncoder.finish()))
            surface!!.present()
        }
    }

    fun render(
        commandEncoder: GPUCommandEncoder,
        image: Image,
        level: Int,
        dst: GPUTexture,
        x: Float,
        y: Float,
        scale: Float
    ) {
        val mipmap = image.mipmaps[level]

        val vx = round(((-x * width / mipmap.width + 0.5) * mipmap.width)).toInt()
        val vy = round(((-y * height / mipmap.height + 0.5) * mipmap.height)).toInt()

        val quad = mipmap.getQuad(vx, vy)

        image.byteBuffer.putFloat(
            0,
            (0.5f / scale + x) * mipmap.scale + (quad.x - 0.5f * mipmap.width) / dst.width
        )
        image.byteBuffer.putFloat(
            4,
            (0.5f / scale + y) * mipmap.scale + (quad.y - 0.5f * mipmap.height) / dst.height
        )
        image.byteBuffer.putFloat(8, scale / mipmap.scale)
        image.byteBuffer.putFloat(12, mipmap.tilesize.toFloat())
        image.byteBuffer.putFloat(16, mipmap.tilesCols.toFloat())
        image.byteBuffer.putFloat(20, mipmap.tilesRows.toFloat())
        image.byteBuffer.putFloat(24, dst.width.toFloat())
        image.byteBuffer.putFloat(28, dst.height.toFloat())
        webgpu.device!!.queue.writeBuffer(image.buffer, 0, image.byteBuffer)

        val pass = commandEncoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(), loadOp = LoadOp.Load, storeOp = StoreOp.Store,
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
                            buffer = image.buffer
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

    fun cleanup() {
        animationJob?.cancel()
        surface?.close()
        images.forEach { it.cleanup() }
        images.clear()
    }

    fun addImage(image: Image) {
        images.add(image)
    }

    fun reset(scope: CoroutineScope, origin: Offset) {
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val targetScale = startScale.coerceIn(minScale, maxScale)
        val max_x = maxX(targetScale)
        val max_y = maxY(targetScale)

        var px: Float
        var py: Float

        if (targetScale != startScale) {
            val diff = 1 / targetScale - 1 / scale
            var x = x + (origin.x - 0.5f) * diff
            var y = y + (origin.y - 0.5f) * diff
            x = x.coerceIn(-max_x, max_x)
            y = y.coerceIn(-max_y, max_y)
            px = (x - startX) / diff
            py = (y - startY) / diff
        } else {
            px = (x.coerceIn(-max_x, max_x) - startX)
            py = (y.coerceIn(-max_y, max_y) - startY)
        }

        animationJob = scope.launch {
            animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                scale = startScale + (targetScale - startScale) * value
                val diff = if (scale != startScale) {
                    1 / scale - 1 / startScale
                } else {
                    value
                }
                var x = (startX + px * diff).orZero()
                var y = (startY + py * diff).orZero()

                if (abs(x) < 1.0e-7) {
                    x = 0f
                }
                if (abs(y) < 1.0e-7) {
                    y = 0f
                }

                setPos(x, y)
            }
        }
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
