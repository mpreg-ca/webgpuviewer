package ca.mpreg.webgpuviewer

import android.content.res.Resources
import android.graphics.Rect
import android.view.View
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.VectorConverter
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.exponentialDecay
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
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerEventTimeoutCancellationException
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.input.pointer.util.addPointerInputChange
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.util.fastCoerceIn
import ca.mpreg.webgpuviewer.WebGpuRenderer.Companion.dispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class WebGpuImageViewerSingleState {
    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var animationJob: Job? = null

    var image: Image? = null

    val width get() = renderer?.width ?: 0
    val height get() = renderer?.height ?: 0

    val imageWidth get() = image?.width ?: 0
    val imageHeight get() = image?.height ?: 0

    var trim: Rect? = null

    val homeScale get() = trim?.let { getMinScale(it.width(), it.height()) } ?: minScale

    val homeX: Float
        get() {
            val trim = trim ?: return 0f
            val left = -((trim.left - 0.5f * imageWidth) / trim.width() + 0.5f)
            val maxX = maxX(homeScale)
            return (left / homeScale).fastCoerceIn(-maxX, maxX)
        }

    val homeY: Float
        get() {
            val trim = trim ?: return 0f
            val top = -((trim.top - 0.5f * imageHeight) / trim.height() + 0.5f)
            val maxY = maxY(homeScale)
            return (top / homeScale).fastCoerceIn(-maxY, maxY)
        }

    val atHome get() = x == homeX && y == homeY && scale == homeScale

    val minScale
        get() = if (imageWidth == 0 || imageHeight == 0) {
            1f
        } else {
            getMinScale(imageWidth, imageHeight)
        }

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    val doubleTapScale get() = max(dpi, minScale * 2)

    val maxScale get() = max(doubleTapScale * 2, 2f)

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

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (this.x == x && this.y == y && this.scale == scale) {
            return
        }
        this.x = x
        this.y = y
        this.scale = scale

        render()
    }

    var renderer: WebGpuRenderer? = null
    var scope: CoroutineScope? = null

    @Synchronized
    fun init(scope: CoroutineScope, renderer: WebGpuRenderer) {
        this.scope = scope
        this.renderer = renderer

        scope.launch {
            _postInit.forEach { it() }
            _postInit.clear()
        }
    }

    fun render() {
        renderer?.render { encoder, texture ->
            image?.let {
                ImageShaderBasic.render(it, encoder, texture, x, y, scale)
            }
        }
    }

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

        val maxX = maxX(targetScale)
        val maxY = maxY(targetScale)

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
fun WebGpuImageViewerSingle(
    modifier: Modifier = Modifier,
    state: WebGpuImageViewerSingleState,
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
                    state.animateTo(Offset(0.5f, 0.5f))

                    if (state.scale > state.minScale) {
                        view.parent?.requestDisallowInterceptTouchEvent(true)
                    }

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        val secondDown = waitForDown(doubleTapTimeout) ?: return@awaitEachGesture
                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // double tap
                            if (state.atHome) {
                                val origin = Offset(
                                    secondDown.position.x / state.width,
                                    secondDown.position.y / state.height
                                )
                                state.animateTo(origin, targetScale = state.doubleTapScale)
                            } else {
                                state.home()
                            }
                        } else {
                            // double tap drag
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)

                            val dragPointerId = secondDown.id

                            val originalScale = state.scale
                            val originalX = state.x
                            val originalY = state.y
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
                                    view.parent?.requestDisallowInterceptTouchEvent(true)

                                    val px = secondDown.position.x / state.width - 0.5f
                                    val py = secondDown.position.y / state.height - 0.5f

                                    val newScale =
                                        originalScale * 10f.pow(2 * totalDeltaY / state.height)

                                    state.scale = newScale
                                    val diff = 1 / state.scale - 1 / originalScale

                                    state.setPos(
                                        (originalX + px * diff).orZero(),
                                        (originalY + py * diff).orZero()
                                    )

                                    change.consume()
                                }
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 200 && state.scale > state.homeScale && state.scale < state.maxScale) {
                                // fling zoom
                                state.animationJob = scope.launch {
                                    val animation = Animatable(0f)
                                    animation.snapTo(0f)
                                    animation.animateDecay(velocity.y, exponentialDecay()) {
                                        val px = secondDown.position.x / state.width - 0.5f
                                        val py = secondDown.position.y / state.height - 0.5f

                                        val newScale =
                                            originalScale * 10f.pow(2 * (totalDeltaY + value) / state.height)

                                        state.scale =
                                            newScale.fastCoerceIn(state.homeScale, state.maxScale)
                                        val diff = 1 / state.scale - 1 / originalScale

                                        val x = (originalX + px * diff).orZero()
                                        val y = (originalY + py * diff).orZero()

                                        val maxX = state.maxX()
                                        val maxY = state.maxY()

                                        state.setPos(
                                            x.fastCoerceIn(-maxX, maxX), y.fastCoerceIn(-maxY, maxY)
                                        )
                                    }
                                }
                            } else {
                                state.animateTo(
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

                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

                        view.parent?.requestDisallowInterceptTouchEvent(true)

                        state.animationJob?.cancel()

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
                                view.parent?.requestDisallowInterceptTouchEvent(true)

                                if (event.changes.size > 1 && event.changes.all { it.pressed }) {
                                    view.parent?.requestDisallowInterceptTouchEvent(true)
                                    if (single) {
                                        velocityTracker.resetTracking()
                                        acc = Offset.Zero
                                    }
                                    velocityTracker.addPointerInputChange(change)
                                    single = false

                                    scaleOrigin = Offset(
                                        centroid.x / state.width, centroid.y / state.height
                                    )
                                } else if (single) {
                                    velocityTracker.addPointerInputChange(change)
                                }

                                val zoom = event.calculateZoom()
                                val pan = event.calculatePan()
                                acc += pan

                                if (zoom != 1f || pan != Offset.Zero) {
                                    val newScale = state.scale * zoom
                                    val diff = 1 / newScale - 1 / state.scale

                                    var x = state.x + (pan.x / state.width) / state.scale
                                    var y = state.y + (pan.y / state.height) / state.scale

                                    x += (centroid.x / state.width - 0.5f) * diff
                                    y += (centroid.y / state.height - 0.5f) * diff

                                    val maxX = state.maxX(newScale)
                                    val maxY = state.maxY(newScale)

                                    if (state.scale != newScale || (!view.scrollable(true) && !view.scrollable(
                                            false
                                        )) || (view.scrollable(true) && x.fastCoerceIn(
                                            -maxX, maxX
                                        ) != state.x) || (view.scrollable(false) && y.fastCoerceIn(
                                            -maxY, maxY
                                        ) != state.y)
                                    ) {
                                        state.scale = newScale

                                        if (single) {
                                            x = x.fastCoerceIn(-maxX, maxX)
                                            y = y.fastCoerceIn(-maxY, maxY)
                                        }

                                        state.setPos(x.orZero(), y.orZero())

                                        view.parent?.requestDisallowInterceptTouchEvent(true)
                                        event.changes.forEach {
                                            if (it.positionChanged()) {
                                                it.consume()
                                            }
                                        }
                                    } else if (single) {
                                        view.parent?.requestDisallowInterceptTouchEvent(false)
                                    }
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        val maxX = state.maxX()
                        val maxY = state.maxY()

                        val velocity = velocityTracker.calculateVelocity()
                        if ((state.scale >= state.homeScale) && (state.scale <= state.maxScale) && (lastEventTime - lastMoveTime) < 100 && (abs(
                                velocity.x
                            ) > 400 || abs(velocity.y) > 400) && (state.x.fastCoerceIn(
                                -maxX, maxX
                            ) == state.x || state.y.fastCoerceIn(-maxY, maxY) == state.y)
                        ) {
                            // fling pan
                            state.animationJob = scope.launch {
                                fling.snapTo(Offset.Zero)
                                var lastOffset = Offset.Zero
                                fling.animateDecay(
                                    Offset(velocity.x, velocity.y), exponentialDecay()
                                ) {
                                    val delta = value - lastOffset
                                    lastOffset = value
                                    val dx = (delta.x / state.width) / state.scale
                                    val dy = (delta.y / state.height) / state.scale
                                    state.setPos(
                                        (state.x + dx).fastCoerceIn(-maxX, maxX).orZero(),
                                        (state.y + dy).fastCoerceIn(-maxY, maxY).orZero()
                                    )
                                }
                            }
                        } else {
                            state.animateTo(scaleOrigin)
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

fun View.scrollable(horizontal: Boolean): Boolean {
    var p = parent
    while (p != null && p is View) {
        if (horizontal && (p.canScrollHorizontally(1) || p.canScrollHorizontally(-1))) return true
        if (!horizontal && (p.canScrollVertically(1) || p.canScrollVertically(-1))) return true
        p = p.parent
    }
    return false
}

suspend fun AwaitPointerEventScope.waitForCleanUp(
    pointerId: PointerId, timeout: Long, touchSlop: Float
): PointerEvent? = try {
    withTimeout(timeout) {
        var acc = Offset.Zero

        while (true) {
            val event = awaitPointerEvent()
            val change = event.changes.firstOrNull { it.id == pointerId } ?: return@withTimeout null
            acc += event.calculatePan()
            if (acc.getDistance() > touchSlop) {
                return@withTimeout null
            }
            if (change.changedToUp()) {
                return@withTimeout event
            }
        }
    }
} catch (e: PointerEventTimeoutCancellationException) {
    null
} as PointerEvent?

suspend fun AwaitPointerEventScope.waitForDown(timeout: Long) = try {
    withTimeout(timeout) {
        var down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        while (down == null) {
            down = awaitPointerEvent().changes.firstOrNull { it.pressed }
        }
        down
    }
} catch (e: PointerEventTimeoutCancellationException) {
    null
}

fun Float.orZero(): Float = if (this.isNaN()) 0f else this
