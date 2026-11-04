package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.ErrorType
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.InternalException
import androidx.webgpu.OutOfMemoryException
import androidx.webgpu.ValidationException
import androidx.webgpu.WebGpuRuntimeException
import ca.mpreg.webgpuviewer.renderer.ErrorScopes.pump
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.asExecutor
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.Duration.Companion.milliseconds

/** Error scopes popped without suspending. Their answers need events processed: see [pump]. */
internal object ErrorScopes {
    private val pending = AtomicInteger()

    /** Pops [device]'s innermost scope; [onError] gets the error type, on the render thread, if it caught one. */
    fun pop(device: GPUDevice, onError: (type: Int) -> Unit) {
        pending.incrementAndGet()
        device.popErrorScope(
            WebGpuRenderer.dispatcher.asExecutor(),
            object : GPURequestCallback<Int> {
                override fun onResult(result: Int) {
                    pending.decrementAndGet()
                    if (result != ErrorType.NoError) onError(result)
                }

                // Caught errors arrive here, not in onResult; anything else is a lost device failing
                // every pending pop.
                override fun onError(exception: Exception) {
                    pending.decrementAndGet()
                    if (exception is WebGpuRuntimeException) onError(errorType(exception))
                }
            })
    }

    private fun errorType(e: WebGpuRuntimeException): Int = when (e) {
        is ValidationException -> ErrorType.Validation
        is OutOfMemoryException -> ErrorType.OutOfMemory
        is InternalException -> ErrorType.Internal
        else -> ErrorType.Unknown
    }

    /** The innermost scope's error message, or null. Pops at once; the answer needs events processed. */
    fun popMessage(device: GPUDevice): Deferred<String?> {
        val answer = CompletableDeferred<String?>()
        device.popErrorScope(Executor(Runnable::run), object : GPURequestCallback<Int> {
            override fun onResult(result: Int) {
                answer.complete(if (result == ErrorType.NoError) null else ErrorType.toString(result))
            }

            override fun onError(exception: Exception) {
                // Else a lost device, which fails the work itself.
                answer.complete((exception as? WebGpuRuntimeException)?.message)
            }
        })
        return answer
    }

    /** Async WebGPU calls resolve only while events are processed. */
    suspend fun <T> pumpingEvents(block: suspend () -> T): T = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1.milliseconds)
            }
        }
        try {
            block()
        } finally {
            pump.cancel()
        }
    }

    /** Lets pending pops answer. Once a frame, on the render thread. */
    fun pump() {
        if (pending.get() > 0 && WebGpuRenderer.isAvailable) WebGpuRenderer.instance.processEvents()
    }
}
