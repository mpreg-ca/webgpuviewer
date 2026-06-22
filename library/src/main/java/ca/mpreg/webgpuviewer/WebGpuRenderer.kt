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
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.floor
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

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var animationJob: Job? = null

    var images: MutableList<Image> = mutableListOf()

    val imageLeft: Int
        get() = images.map { floor(it.x - it.width / 2.0) }.minOrNull()?.toInt() ?: 0

    val imageRight: Int
        get() = images.map { ceil(it.x + it.width / 2.0) }.minOrNull()?.toInt() ?: 0

    val imageTop: Int
        get() = images.map { floor(it.y - it.height / 2.0) }.minOrNull()?.toInt() ?: 0

    val imageBottom: Int
        get() = images.map { ceil(it.y + it.height / 2.0) }.minOrNull()?.toInt() ?: 0

    val imageWidth: Int
        get() = imageRight - imageLeft

    val imageHeight: Int
        get() = imageBottom - imageTop

    var homeScale = 1f
    var homeX = 0f
    var homeY = 0f

    val atHome: Boolean
        get() = x == homeX && y == homeY && scale == homeScale

    fun getHomeX(left: Float, scale: Float): Float {
        val maxX = maxX(scale)
        return (left / scale).coerceIn(-maxX, maxX)
    }

    fun getHomeY(top: Float, scale: Float): Float {
        val maxY = maxY(scale)
        return (top / scale).coerceIn(-maxY, maxY)
    }

    val minScale: Float
        get() = if (surface == null || imageWidth == 0 || imageHeight == 0) {
            1f
        } else {
            getMinScale(imageWidth, imageHeight)
        }

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    val doubleTapScale: Float
        get() = max(dpi, minScale * 2)

    val maxScale: Float
        get() = max(doubleTapScale * 2, 2f)

    fun getMinScale(width: Int, height: Int): Float {
        val ratioX = this.width.toFloat() / width.toFloat()
        val ratioY = this.height.toFloat() / height.toFloat()
        return max(0.01f, min(ratioX, ratioY))
    }

    fun maxX(scale: Float = this.scale): Float {
        return max(0f, (imageWidth.toFloat() / width - 1 / scale) / 2)
    }

    fun maxY(scale: Float = this.scale): Float {
        return max(0f, (imageHeight.toFloat() / height - 1 / scale) / 2)
    }

    private lateinit var frameClock: MonotonicFrameClock

    private var scope: CoroutineScope? = null

    private val _postInit = mutableListOf<(() -> Unit)>()

    @Synchronized
    fun post(fn: () -> Unit) {
        if (scope?.isActive == true) {
            fn()
        } else {
            _postInit.add(fn)
        }
    }

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.frameClock = scope.coroutineContext[MonotonicFrameClock]
            ?: error("No MonotonicFrameClock found in Composable context")

        this.scope = scope
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

        this._postInit.forEach { it() }
    }

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (this.x == x && this.y == y && this.scale == scale) {
            return
        }
        this.x = x
        this.y = y
        this.scale = scale
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
    }

    fun home() {
        animateTo(targetScale = homeScale)
    }

    fun animateTo(
        origin: Offset? = null,
        targetX: Float = homeX,
        targetY: Float = homeY,
        targetScale: Float = scale
    ) {
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val targetScale = targetScale.coerceIn(minScale, maxScale)

        val max_x = maxX(targetScale)
        val max_y = maxY(targetScale)

        val diffEnd = if (targetScale != startScale) {
            1 / targetScale - 1 / scale
        } else {
            1f
        }

        val endX = if (origin != null) {
            if (targetScale != startScale) {
                (startX + (origin.x - 0.5f) * diffEnd).coerceIn(-max_x, max_x)
            } else {
                x.coerceIn(-max_x, max_x)
            }
        } else {
            targetX
        }

        val endY = if (origin != null) {
            if (targetScale != startScale) {
                (startY + (origin.y - 0.5f) * diffEnd).coerceIn(-max_y, max_y)
            } else {
                y.coerceIn(-max_y, max_y)
            }
        } else {
            targetY
        }

        animationJob = scope?.launch {
            animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                val scale = startScale * (1 - value) + targetScale * value
                val c = if (startScale != targetScale) {
                    val diff = 1 / scale - 1 / startScale
                    (diff / diffEnd).coerceIn(0f, 1f)
                } else {
                    value
                }

                setPos(
                    (startX * (1 - c) + endX * c).orZero(),
                    (startY * (1 - c) + endY * c).orZero(),
                    scale
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
