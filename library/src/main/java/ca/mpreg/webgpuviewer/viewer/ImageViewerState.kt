package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.view.Surface
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
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
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.min

open class ImageViewerState(var isVertical: Boolean = false) {
    val renderer = WebGpuRenderer()

    var animationJob: Job? = null

    val width get() = renderer.width
    val height get() = renderer.height

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    var scope: CoroutineScope? = null

    /** Top padding in pixels to avoid display cutout. Set automatically by ImageViewer when avoidCutout is true. */
    var cutoutTopPx: Float = 0f
        set(value) {
            if (field != value) {
                field = value
                // Move current page to new home position when cutout changes
                getPage(0)?.home()
            }
        }

    /** When true, images will be positioned/scaled to avoid the display cutout. */
    var avoidCutout: Boolean by mutableStateOf(false)

    /** When true, always shift images below cutout. When false, only shift if image would overlap cutout. */
    var alwaysAvoidCutout: Boolean by mutableStateOf(false)

    fun getMinScale(width: Int, height: Int): Float {
        val ratioX = this.width.toFloat() / width.toFloat()
        val ratioY = this.height.toFloat() / height.toFloat()
        return max(0.01f, min(ratioX, ratioY))
    }

    fun maxX(width: Int, scale: Float): Float {
        return max(0f, (width.toFloat() / this.width - 1 / scale) / 2)
    }

    fun minY(height: Int, scale: Float): Float {
        return -max(0f, (height.toFloat() / this.height - 1 / scale) / 2)
    }

    fun minY(height: Int, scale: Float, homeY: Float): Float {
        return min(minY(height, scale), homeY)
    }

    fun maxY(height: Int, scale: Float): Float {
        return max(0f, (height.toFloat() / this.height - 1 / scale) / 2)
    }

    fun maxY(height: Int, scale: Float, homeY: Float): Float {
        return max(maxY(height, scale), homeY)
    }

    private var suppressPageChange = false

    var pageOffset = 0f
        set(value) {
            var v = value
            var pageDelta = 0

            if (!suppressPageChange) {
                while (v >= 1f && haveNext) {
                    pageDelta += 1
                    v -= 1f
                }
                while (v <= -1f && havePrev) {
                    pageDelta -= 1
                    v += 1f
                }
            }

            if (!haveNext) v = v.fastCoerceAtMost(1f)
            if (!havePrev) v = v.fastCoerceAtLeast(-1f)

            field = v

            if (pageDelta != 0) {
                onPageChange?.invoke(pageDelta)
            }
        }

    private fun setPageOffsetDirect(value: Float) {
        suppressPageChange = true
        pageOffset = value
        suppressPageChange = false
    }

    fun animatePageTurn(direction: Int) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            setPageOffsetDirect(direction.toFloat())
            invalidate()
            try {
                Animatable(direction.toFloat()).animateTo(
                    0f, animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
                ) {
                    setPageOffsetDirect(value)
                    invalidate()
                }
            } finally {
                // Ensure we snap to exactly 0 and clear override
                setPageOffsetDirect(0f)
                transitionFromPage = null
                invalidate()
            }
        }
    }

    val havePrev get() = getPage(-1) != null
    val haveNext get() = getPage(1) != null

    var fetchPage: ((Int) -> ImagePage?)? = null

    var onPageChange: ((Int) -> Unit)? = null
    var onTap: ((Offset) -> Unit)? = null
    var onLongTap: ((Offset) -> Unit)? = null

    /** Override for the "from" page during far navigation animation */
    var transitionFromPage: ImagePage? = null

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
        replay = 1, extraBufferCapacity = 1, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    fun invalidate() {
        renderFlow.tryEmit(0)
    }

    suspend fun collect() {
        renderFlow.collectLatest {
            // Capture render state on main thread before any thread switching
            val snapshot = captureRenderState() ?: return@collectLatest
            // Now render on GPU thread with captured state
            withContext(dispatcher) {
                renderer.render { encoder, texture ->
                    renderSnapshot(encoder, texture, snapshot)
                }
            }
        }
    }

    protected open fun captureRenderState(): Any? {
        val currentPage = getPage(0) ?: return null
        val offset = pageOffset
        val adjacentPage = when {
            offset == 0f -> null
            // Use override if set (for far navigation)
            transitionFromPage != null -> transitionFromPage
            offset > 0f -> getPage(1)
            else -> getPage(-1)
        }
        return RenderSnapshot(currentPage, adjacentPage, offset, transition, firstPos, currentPos)
    }

    private class RenderSnapshot(
        val currentPage: ImagePage,
        val adjacentPage: ImagePage?,
        val offset: Float,
        val transition: Transition,
        val firstPos: Offset,
        val currentPos: Offset
    )

    protected open suspend fun renderSnapshot(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        snapshot: Any
    ) {
        val s = snapshot as RenderSnapshot
        if (s.adjacentPage != null && s.offset != 0f) {
            s.transition.render(
                s.currentPage,
                s.adjacentPage,
                encoder,
                texture,
                s.offset,
                s.firstPos,
                s.currentPos
            )
        } else {
            TransitionBasic.render(s.currentPage, encoder, texture, 0f, 0f, 1f)
        }
    }

    private val _postInit = mutableListOf<(suspend () -> Unit)>()

    @Synchronized
    fun post(fn: suspend () -> Unit) {
        val activeScope = scope
        if (activeScope?.isActive == true) {
            activeScope.launch(dispatcher) {
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