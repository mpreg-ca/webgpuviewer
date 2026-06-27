package ca.mpreg.webgpuviewer

import android.content.res.Resources
import android.graphics.Rect
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.exponentialDecay
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.AndroidExternalSurface
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.input.pointer.util.addPointerInputChange
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.util.fastCoerceAtLeast
import androidx.compose.ui.util.fastCoerceAtMost
import androidx.compose.ui.util.fastCoerceIn
import ca.mpreg.webgpuviewer.WebGpuRenderer.Companion.dispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class WebGpuImageViewerPage(val image: Image) {
    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var animationJob: Job? = null

    val width get() = image.width
    val height get() = image.height

    var trim: Rect? = null

    val homeScale get() = trim?.let { parent?.getMinScale(it.width(), it.height()) } ?: minScale

    val homeX: Float
        get() {
            val trim = trim ?: return 0f
            val parent = parent ?: return 0f
            val left = -((trim.left - 0.5f * width) / trim.width() + 0.5f)
            val maxX = parent.maxX(width, homeScale)
            return (left / homeScale).fastCoerceIn(-maxX, maxX)
        }

    val homeY: Float
        get() {
            val trim = trim ?: return 0f
            val parent = parent ?: return 0f
            val top = -((trim.top - 0.5f * height) / trim.height() + 0.5f)
            val maxY = parent.maxY(height, homeScale)
            return (top / homeScale).fastCoerceIn(-maxY, maxY)
        }

    val atHome get() = x == homeX && y == homeY && scale == homeScale

    var onInvalidate: (() -> Unit)? = null

    var parent: WebGpuImageViewerState? = null

    val minScale get() = parent?.getMinScale(width, height) ?: 1f

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    val doubleTapScale get() = max(dpi, minScale * 2)

    val maxScale get() = max(doubleTapScale * 2, 2f)

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (this.x == x && this.y == y && this.scale == scale) {
            return
        }

        this.x = x
        this.y = y
        this.scale = scale

        onInvalidate?.invoke()
    }

    var scope: CoroutineScope? = null

    fun home() {
        animateTo(targetScale = homeScale)
    }

    fun animateTo(
        origin: Offset? = null,
        targetX: Float = homeX,
        targetY: Float = homeY,
        targetScale: Float = scale,
    ) {
        animationJob?.cancel()

        val startScale = scale
        val startX = x
        val startY = y

        val targetScale = targetScale.fastCoerceIn(minScale, maxScale)

        val maxX = parent?.maxX(width, targetScale) ?: 0f
        val maxY = parent?.maxY(height, targetScale) ?: 0f

        val diffEnd = if (targetScale != startScale) {
            1 / targetScale - 1 / scale
        } else {
            1f
        }

        val endX = if (origin != null) {
            if (targetScale != startScale) {
                (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(-maxX, maxX)
            } else {
                x.fastCoerceIn(-maxX, maxX)
            }
        } else {
            targetX
        }

        val endY = if (origin != null) {
            if (targetScale != startScale) {
                (startY + (origin.y - 0.5f) * diffEnd).fastCoerceIn(-maxY, maxY)
            } else {
                y.fastCoerceIn(-maxY, maxY)
            }
        } else {
            targetY
        }

        animationJob = scope?.launch {
            animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                val scale = startScale * (1 - value) + targetScale * value
                val c = if (startScale != targetScale) {
                    val diff = 1 / scale - 1 / startScale
                    (diff / diffEnd).fastCoerceIn(0f, 1f)
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

class WebGpuImageViewerState {
    var animationJob: Job? = null

    val width get() = renderer?.width ?: 0
    val height get() = renderer?.height ?: 0

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    var renderer: WebGpuRenderer? = null
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

    var pageCount = 0

    var pageOffset = 0f
        set(value) {
            field = value.fastCoerceIn(0f, (pageCount - 1).toFloat().fastCoerceAtLeast(0f))
        }

    var currentPageIndex: Int
        get() = kotlin.math.round(pageOffset).toInt()
            .fastCoerceIn(0, (pageCount - 1).fastCoerceAtLeast(0))
        set(value) {
            pageOffset = value.toFloat()
        }

    var fetchPage: (suspend (Int) -> WebGpuImageViewerPage)? = null

    val pages = mutableMapOf<Int, WebGpuImageViewerPage?>()

    private val mutex = Mutex()

    suspend fun getPage(index: Int): WebGpuImageViewerPage? {
        if (index >= pageCount) return null
        if (index < 0) return null

        mutex.withLock {
            pages[index] = pages[index] ?: fetchPage?.invoke(index)?.apply {
                parent = this@WebGpuImageViewerState
                scope = this@WebGpuImageViewerState.scope
                onInvalidate = {
                    render()
                }
                setPos(homeX, homeY, homeScale)
            }

            return pages[index]
        }
    }

    @Synchronized
    fun init(scope: CoroutineScope, renderer: WebGpuRenderer) {
        this.scope = scope
        this.renderer = renderer

        scope.launch {
            _postInit.forEach { it() }
            _postInit.clear()
        }
    }

    var firstDownPos = Offset.Zero
    var currentTouchPos = Offset.Zero

    fun render() {
        renderer?.render { encoder, texture ->
            val idx = kotlin.math.floor(pageOffset).toInt()
            val frac = pageOffset - idx

            val currentPage = getPage(idx) ?: return@render

            if (frac == 0f) {
                ImageShaderBasic.render(currentPage, encoder, texture, 0f, 0f, 1f)
            } else {
                getPage(idx + 1)?.let {
                    ImageShaderBasic.render(
                        currentPage, it, encoder, texture, frac, firstDownPos, currentTouchPos
                    )
//                    ImageShaderFlipRight.render(
//                        currentPage, it, encoder, texture, frac, firstDownPos, currentTouchPos
//                    )
//                    ImageShaderFlipLeft.render(
//                        currentPage, it, encoder, texture, frac, firstDownPos, currentTouchPos
//                    )
                } ?: ImageShaderBasic.render(currentPage, encoder, texture, 0f, 0f, 1f)
            }
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
    }
}

@Composable
fun WebGpuImageViewer(
    modifier: Modifier = Modifier,
    state: WebGpuImageViewerState,
) {
    val scope = rememberCoroutineScope()

    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    val renderer = remember { WebGpuRenderer() }

    val view = LocalView.current

    AndroidExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    state.animationJob?.cancel()
                    state.pageOffset = state.currentPageIndex.toFloat()
                    val page = runBlocking(dispatcher) { state.getPage(state.currentPageIndex) }
                        ?: return@awaitEachGesture
                    page.animateTo(Offset(0.5f, 0.5f))

                    view.parent?.requestDisallowInterceptTouchEvent(true)

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        val secondDown = waitForDown(doubleTapTimeout) ?: return@awaitEachGesture
                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // double tap
                            if (page.atHome) {
                                val origin = Offset(
                                    secondDown.position.x / state.width,
                                    secondDown.position.y / state.height
                                )
                                page.animateTo(origin, targetScale = page.doubleTapScale)
                            } else {
                                page.home()
                            }
                        } else {
                            // double tap drag
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)

                            val dragPointerId = secondDown.id

                            val originalScale = page.scale
                            val originalX = page.x
                            val originalY = page.y
                            var totalDeltaY = 0f

                            state.animationJob?.cancel()

                            while (true) {
                                val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                val change =
                                    event.changes.firstOrNull { it.id == dragPointerId && it.positionChanged() }

                                if (change == null || change.changedToUp() || change.isConsumed) {
                                    break
                                }

                                velocityTracker.addPointerInputChange(change)

                                val pan = event.calculatePan()
                                totalDeltaY += pan.y
                                if (totalDeltaY != 0f) {

                                    val px = secondDown.position.x / state.width - 0.5f
                                    val py = secondDown.position.y / state.height - 0.5f

                                    val newScale =
                                        originalScale * 10f.pow(2 * totalDeltaY / state.height)

                                    page.scale = newScale
                                    val diff = 1 / page.scale - 1 / originalScale

                                    page.setPos(
                                        (originalX + px * diff).orZero(),
                                        (originalY + py * diff).orZero()
                                    )

                                    change.consume()
                                }
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 200 && page.scale > page.homeScale && page.scale < page.maxScale) {
                                // fling zoom
                                state.animationJob = scope.launch {
                                    val animation = Animatable(0f)
                                    animation.snapTo(0f)
                                    animation.animateDecay(velocity.y, exponentialDecay()) {
                                        val px = secondDown.position.x / state.width - 0.5f
                                        val py = secondDown.position.y / state.height - 0.5f

                                        val newScale =
                                            originalScale * 10f.pow(2 * (totalDeltaY + value) / state.height)

                                        page.scale =
                                            newScale.fastCoerceIn(page.homeScale, page.maxScale)
                                        val diff = 1 / page.scale - 1 / originalScale

                                        val x = (originalX + px * diff).orZero()
                                        val y = (originalY + py * diff).orZero()

                                        val maxX = state.maxX(page.width, page.scale)
                                        val maxY = state.maxY(page.height, page.scale)

                                        page.setPos(
                                            x.fastCoerceIn(-maxX, maxX), y.fastCoerceIn(-maxY, maxY)
                                        )
                                    }
                                }
                            } else {
                                page.animateTo(
                                    Offset(
                                        secondDown.position.x / state.width,
                                        secondDown.position.y / state.height
                                    )
                                )
                            }
                        }
                    } else {
                        var lastMoveTime = firstDown.uptimeMillis
                        var lastEventTime: Long = firstDown.uptimeMillis
                        var acc = Offset.Zero

                        var scaleOrigin = Offset(0.5f, 0.5f)

                        var single = true
                        var pageTurning = false

                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

                        page.animationJob?.cancel()

                        do {
                            val event = awaitPointerEvent()
                            val canceled = event.changes.any { it.isConsumed }
                            if (!canceled) {
                                val change = event.changes[0]
                                lastEventTime = change.uptimeMillis

                                if (change.positionChanged()) {
                                    lastMoveTime = change.uptimeMillis
                                }

                                val centroid = event.calculateCentroid(useCurrent = true)

                                var pointerCountChanged = false
                                if (event.changes.size > 1 && event.changes.all { it.pressed }) {
                                    if (single && !pageTurning) {
                                        velocityTracker.resetTracking()
                                        acc = Offset.Zero
                                        pointerCountChanged = true
                                    }
                                    if (!pageTurning) {
                                        velocityTracker.addPointerInputChange(change)
                                        single = false
                                        scaleOrigin = Offset(
                                            centroid.x / state.width, centroid.y / state.height
                                        )
                                    }
                                } else if (single) {
                                    velocityTracker.addPointerInputChange(change)
                                }

                                if (pointerCountChanged) continue

                                val pan = event.calculatePan()
                                acc += pan

                                if (pageTurning) {
                                    state.pageOffset += -pan.x / state.width
                                    state.currentTouchPos = event.changes[0].position
                                    state.render()
                                    event.changes.forEach { if (it.positionChanged()) it.consume() }
                                } else {
                                    val zoom = event.calculateZoom()

                                    if (zoom != 1f || pan != Offset.Zero) {
                                        val newScale = page.scale * zoom
                                        val diff = 1 / newScale - 1 / page.scale

                                        var x = page.x + (pan.x / state.width) / page.scale
                                        var y = page.y + (pan.y / state.height) / page.scale

                                        x += (centroid.x / state.width - 0.5f) * diff
                                        y += (centroid.y / state.height - 0.5f) * diff

                                        val maxX = state.maxX(page.width, newScale)
                                        val maxY = state.maxY(page.height, newScale)

                                        if (single) {
                                            val clampedX = x.fastCoerceIn(-maxX, maxX)
                                            val overflow = x - clampedX
                                            if (overflow != 0f && abs(acc.x) > abs(acc.y)) {
                                                page.animateTo(Offset(0.5f, 0.5f))
                                                pageTurning = true
                                                state.firstDownPos = firstDown.position
                                                state.pageOffset += -overflow * page.scale
                                                state.render()
                                            }
                                            x = clampedX
                                            y = y.fastCoerceIn(-maxY, maxY)
                                        }

                                        page.scale = newScale
                                        page.setPos(x.orZero(), y.orZero())

                                        event.changes.forEach { if (it.positionChanged()) it.consume() }
                                    }
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        if (pageTurning) {
                            val velocity = velocityTracker.calculateVelocity()
                            val initialVelocity = -velocity.x / state.width
                            val nearest = kotlin.math.round(state.pageOffset)

                            val target = when {
                                initialVelocity > 1f -> kotlin.math.floor(state.pageOffset + 1f)
                                    .fastCoerceAtMost((state.pageCount - 1).toFloat())

                                initialVelocity < -1f -> kotlin.math.ceil(state.pageOffset - 1f)
                                    .fastCoerceAtLeast(0f)

                                else -> nearest
                            }

                            state.animationJob = scope.launch {
                                val anim = Animatable(state.pageOffset)
                                anim.animateTo(
                                    target,
                                    spring(dampingRatio = 1f, stiffness = 400f),
                                    initialVelocity
                                ) {
                                    state.pageOffset = value
                                    state.render()
                                }
                            }
                        } else {
                            val maxX = state.maxX(page.width, page.scale)
                            val maxY = state.maxY(page.height, page.scale)

                            val velocity = velocityTracker.calculateVelocity()
                            if ((page.scale >= page.homeScale) && (page.scale <= page.maxScale) && (lastEventTime - lastMoveTime) < 100 && (abs(
                                    velocity.x
                                ) > 400 || abs(velocity.y) > 400) && (page.x.fastCoerceIn(
                                    -maxX, maxX
                                ) == page.x || page.y.fastCoerceIn(-maxY, maxY) == page.y)
                            ) {
                                // fling pan
                                page.animationJob = scope.launch {
                                    fling.snapTo(Offset.Zero)
                                    var lastOffset = Offset.Zero
                                    fling.animateDecay(
                                        Offset(velocity.x, velocity.y), exponentialDecay()
                                    ) {
                                        val delta = value - lastOffset
                                        lastOffset = value
                                        val dx = (delta.x / state.width) / page.scale
                                        val dy = (delta.y / state.height) / page.scale
                                        page.setPos(
                                            (page.x + dx).fastCoerceIn(-maxX, maxX).orZero(),
                                            (page.y + dy).fastCoerceIn(-maxY, maxY).orZero()
                                        )
                                    }
                                }
                            } else {
                                page.animateTo(scaleOrigin)
                            }
                        }
                    }
                }
            }, isOpaque = false
    ) {
        onSurface { surface, width, height ->
            try {
                renderer.init(scope, surface, width, height)
                state.init(scope, renderer)

                state.render()
                awaitCancellation()
            } finally {
                state.cleanup()
                renderer.cleanup()
            }
        }
    }
}
