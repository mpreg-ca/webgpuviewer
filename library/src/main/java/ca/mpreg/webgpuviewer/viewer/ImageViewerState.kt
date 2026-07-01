package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.util.Log
import android.view.Surface
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.util.fastCoerceAtLeast
import androidx.compose.ui.util.fastCoerceAtMost
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer.Companion.dispatcher
import ca.mpreg.webgpuviewer.transition.Transition
import ca.mpreg.webgpuviewer.transition.TransitionBasic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.max
import kotlin.math.min

open class ImageViewerState(var isVertical: Boolean = false) {
    val renderer = WebGpuRenderer()

    var animationJob: Job? = null

    val width get() = renderer.width
    val height get() = renderer.height

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    var scope: CoroutineScope? = null

    fun getMinScale(width: Int, height: Int): Float {
        val ratioX = this.width.toFloat() / width.toFloat()
        val ratioY = this.height.toFloat() / height.toFloat()
        return max(0.01f, min(ratioX, ratioY))
    }

    fun maxX(width: Int, scale: Float): Float {
        return max(0f, (width.toFloat() / this.width - 1 / scale) / 2)
    }

    fun maxY(height: Int, scale: Float): Float {
        return max(0f, (height.toFloat() / this.height - 1 / scale) / 2)
    }

    var pageOffset = 0f
        set(value) {
            var v = value
            var pageDelta = 0

            while (v >= 1f && haveNext) {
                pageDelta += 1
                v -= 1f
            }
            while (v <= -1f && havePrev) {
                pageDelta -= 1
                v += 1f
            }

            if (!haveNext) v = v.fastCoerceAtMost(0f)
            if (!havePrev) v = v.fastCoerceAtLeast(0f)

            field = v

            if (pageDelta != 0) {
                onPageChange?.invoke(pageDelta)
            }
        }

    val havePrev get() = getPage(-1) != null
    val haveNext get() = getPage(1) != null

    var fetchPage: ((Int) -> ImagePage?)? = null

    var onPageChange: ((Int) -> Unit)? = null
    var onTap: ((Offset) -> Unit)? = null
    var onLongTap: ((Offset) -> Unit)? = null

    fun getPage(index: Int): ImagePage? {
        return fetchPage?.invoke(index)?.apply {
            parent = this@ImageViewerState
            scope = this@ImageViewerState.scope
            onInvalidate = {
                invalidate()
            }
        }
    }

    @Synchronized
    fun init(scope: CoroutineScope, surface: Surface, width: Int, height: Int) {
        this.renderer.init(scope, surface, width, height)
        this.scope = scope

        scope.launch {
            _postInit.forEach { it() }
            _postInit.clear()
        }
    }

    var firstPos = Offset.Zero
    var currentPos = Offset.Zero

    var transition: Transition = if (isVertical) TransitionBasic.Vertical else TransitionBasic

    var renderFlow = MutableSharedFlow<Int>(
        replay = 0, extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    fun invalidate() {
        scope?.launch {
            renderFlow.emit(0)
        }
    }

    suspend fun collect() {
        Log.i("WebGpuImageViewer", "start collecting")
        renderFlow.collectLatest {
            Log.i("WebGpuImageViewerState", "collect")
            renderer.render { encoder, texture ->
                render(encoder, texture)
            }
        }
    }

    protected open suspend fun render(encoder: GPUCommandEncoder, texture: GPUTexture) {
        val currentPage = getPage(0) ?: return@render

        if (pageOffset == 0f) {
            TransitionBasic.render(currentPage, encoder, texture, 0f, 0f, 1f)
        } else if (pageOffset > 0f) {
            getPage(1)?.let {
                transition.render(
                    currentPage, it, encoder, texture, pageOffset, firstPos, currentPos,
                )
            } ?: TransitionBasic.render(currentPage, encoder, texture, 0f, 0f, 1f)
        } else {
            getPage(-1)?.let {
                transition.render(
                    currentPage, it, encoder, texture, pageOffset, firstPos, currentPos
                )
            } ?: TransitionBasic.render(currentPage, encoder, texture, 0f, 0f, 1f)
        }
    }

    private val _postInit = mutableListOf<(suspend () -> Unit)>()

    @Synchronized
    fun post(fn: suspend () -> Unit) {
        if (scope?.isActive == true) {
            CoroutineScope(dispatcher).launch {
                fn()
            }
        } else {
            _postInit.add(fn)
        }
    }

    fun cleanup() {
        animationJob?.cancel()
        renderer.cleanup()
    }
}