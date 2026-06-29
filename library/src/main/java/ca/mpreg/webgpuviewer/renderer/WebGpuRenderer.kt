package ca.mpreg.webgpuviewer.renderer

import android.view.Surface
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostException
import androidx.webgpu.FeatureLevel
import androidx.webgpu.GPU.createInstance
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPUInstanceDescriptor
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPUSurface
import androidx.webgpu.GPUSurfaceConfiguration
import androidx.webgpu.GPUSurfaceDescriptor
import androidx.webgpu.GPUSurfaceSourceAndroidNativeWindow
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.Util.windowFromSurface
import androidx.webgpu.helper.initLibrary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class WebGpuRenderer {
    companion object {
        var instance: GPUInstance
        var adapter: GPUAdapter
        var device: GPUDevice

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
            }
        }
    }

    @Volatile
    private var surface: GPUSurface? = null

    var width: Int = 0
    var height: Int = 0

    private var scope: CoroutineScope? = null

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.scope = scope
        this.width = width
        this.height = height

        runBlocking(dispatcher) {
            this@WebGpuRenderer.surface = surface.let {
                instance.createSurface(
                    GPUSurfaceDescriptor(
                        surfaceSourceAndroidNativeWindow = GPUSurfaceSourceAndroidNativeWindow(
                            windowFromSurface(it)
                        )
                    )
                ).apply {
                    configure(
                        GPUSurfaceConfiguration(
                            device,
                            width,
                            height,
                            TextureFormat.RGBA8Unorm,
                            TextureUsage.RenderAttachment
                        )
                    )
                }
            }
        }
    }

    private var renderJob: Job? = null
    private var nextRenderJob: (suspend CoroutineScope.() -> Unit)? = null

    private val mutex = Mutex()

    fun render(fn: suspend (GPUCommandEncoder, GPUTexture) -> Unit) {
        runBlocking {
            mutex.withLock {
                this@WebGpuRenderer.nextRenderJob = f@{
                    val surface = surface ?: return@f

                    val texture = surface.getCurrentTexture().texture
                    val encoder = device.createCommandEncoder()

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

                    fn(encoder, texture)

                    device.queue.submit(arrayOf(encoder.finish()))
                    surface.present()

                    mutex.withLock {
                        renderJob = this@WebGpuRenderer.nextRenderJob?.let {
                            scope?.launch(dispatcher, CoroutineStart.DEFAULT, it)
                        }
                        this@WebGpuRenderer.nextRenderJob = null
                    }
                }

                if (renderJob == null) {
                    this@WebGpuRenderer.nextRenderJob?.let {
                        renderJob = scope?.launch(dispatcher, CoroutineStart.DEFAULT, it)
                        this@WebGpuRenderer.nextRenderJob = null
                    }
                }
            }
        }
    }

    fun cleanup() {
        runBlocking(dispatcher) {
            surface?.close()
            surface = null
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
