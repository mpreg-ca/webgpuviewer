package ca.mpreg.webgpuviewer

import android.content.res.Resources
import android.view.Surface
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.tween
import androidx.compose.runtime.MonotonicFrameClock
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.geometry.Offset
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostException
import androidx.webgpu.FeatureLevel
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUColorTargetState
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

object webgpu {
    var instance: GPUInstance
    var adapter: GPUAdapter
    var device: GPUDevice
    var pipeline: GPURenderPipeline
    val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "WebGPU-Render-Thread")
    }.asCoroutineDispatcher()

    init {
        runBlocking {
            initLibrary()

            instance = createInstance(GPUInstanceDescriptor())
            adapter =
                instance.requestAdapter(GPURequestAdapterOptions(featureLevel = FeatureLevel.Compatibility))
            device = adapter.requestDevice(
                GPUDeviceDescriptor(
                    deviceLostCallback = defaultDeviceLostCallback,
                    deviceLostCallbackExecutor = Executor(Runnable::run),
                    uncapturedErrorCallback = defaultUncapturedErrorCallback,
                    uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                )
            )

            val shaderModule = device.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(
                        WebGpuRendererShader.shader
                    )
                )
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
    }
}

class WebGpuRenderer {
    private var surface: GPUSurface? = null

    var width: Int = 0
    var height: Int = 0

    val imageWidth: Int
        get() = if (images.isNotEmpty()) {
            ceil(
                images.map { it.width - it.x }.max() - min(images.map { it.x }.min(), 0f)
            ).toInt()
        } else {
            0
        }
    val imageHeight: Int
        get() = if (images.isNotEmpty()) {
            ceil(
                images.map { it.height - it.y }.max() - min(images.map { it.y }.min(), 0f)
            ).toInt()
        } else {
            0
        }

    var animationJob: Job? = null

    var images: MutableList<Image> = mutableListOf()

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var fitScale = 1f
    var doubleTapScale = Resources.getSystem().displayMetrics.densityDpi / 100f

    val minScale: Float
        get() = if (surface == null || imageWidth == 0 || imageHeight == 0) 1f
        else {
            val ratioX = width.toFloat() / imageWidth.toFloat()
            val ratioY = height.toFloat() / imageHeight.toFloat()
            max(0.01f, min(ratioX, ratioY))
        }

    var maxScale = max(doubleTapScale, 2f)

    lateinit var frameClock: MonotonicFrameClock

    var postInit: () -> Unit = {}

    fun init(surface: Surface, width: Int, height: Int, scope: CoroutineScope) {
        this.frameClock = scope.coroutineContext[MonotonicFrameClock]
            ?: error("No MonotonicFrameClock found in Composable context")

        this.width = width
        this.height = height

        this.surface = surface.let {
            webgpu.instance.createSurface(
                GPUSurfaceDescriptor(
                    surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                        windowFromSurface(it)
                    )
                )
            ).apply {
                configure(
                    GPUSurfaceConfiguration(
                        webgpu.device,
                        width,
                        height,
                        TextureFormat.RGBA8Unorm,
                        TextureUsage.RenderAttachment
                    )
                )
            }
        }

        this.postInit()
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
        val surface = surface ?: return

        CoroutineScope(webgpu.dispatcher + this.frameClock).launch {
            withFrameNanos {
                val texture = surface.getCurrentTexture().texture
                val encoder = webgpu.device.createCommandEncoder()

                encoder.beginRenderPass(
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
                ).end()

                images.forEach {
                    it.render(encoder, texture, x, y, scale)
                }

                webgpu.device.queue.submit(arrayOf(encoder.finish()))
                surface.present()
            }
        }
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

    fun reset(
        scope: CoroutineScope, origin: Offset? = null, targetX: Float = 0f, targetY: Float = 0f,
        targetScale: Float = scale.coerceIn(minScale, maxScale)
    ) {
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val max_x = maxX(targetScale)
        val max_y = maxY(targetScale)

        var px: Float
        var py: Float

        if (origin != null) {
            if (targetScale != startScale) {
                val diff = 1 / targetScale - 1 / scale
                var x = startX + (origin.x - 0.5f) * diff
                var y = startY + (origin.y - 0.5f) * diff
                x = x.coerceIn(-max_x, max_x)
                y = y.coerceIn(-max_y, max_y)
                px = (x - startX) / diff
                py = (y - startY) / diff
            } else {
                px = x.coerceIn(-max_x, max_x) - startX
                py = y.coerceIn(-max_y, max_y) - startY
            }
        } else {
            val diff = if (targetScale != startScale) {
                1 / targetScale - 1 / scale
            } else {
                1f
            }

            px = (targetX - startX) / diff
            py = (targetY - startY) / diff
        }

        animationJob = scope.launch {
            animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                scale = startScale * (1 - value) + targetScale * value
                val diff = if (scale != startScale) {
                    1 / scale - 1 / startScale
                } else {
                    value
                }

                setPos(
                    (startX + px * diff).orZero(),
                    (startY + py * diff).orZero()
                )
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
