package ca.mpreg.webgpuviewer

import android.graphics.Bitmap
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.input.pointer.util.addPointerInputChange
import androidx.compose.ui.platform.LocalView
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

@Composable
fun WebGpuImageViewer(
    modifier: Modifier = Modifier,
    renderer: WebGpuRenderer = WebGpuRenderer(),
    bitmap: Bitmap,
    doubleTapScale: Float = LocalView.current.resources.displayMetrics.densityDpi / 200f,
    maxScale: Float = LocalView.current.resources.displayMetrics.densityDpi / 100f,
    startFitWidth: Boolean = true,
    startFitHeight: Boolean = true,
    zoomWide: Boolean = false,
    zoomStartPosition: Offset = Offset.Zero,
) {
    val scope = rememberCoroutineScope()

    val fitScale = remember { mutableStateOf(1f) }
    val doubleTapScale = remember { mutableStateOf(doubleTapScale) }

    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    val view = LocalView.current

    AndroidExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    renderer.reset(scope, Offset(0.5f, 0.5f))

                    if (renderer.scale > renderer.minScale) {
                        view.parent?.requestDisallowInterceptTouchEvent(true)
                    }

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        val secondDown = waitForDown(doubleTapTimeout) ?: return@awaitEachGesture
                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // double tap
                            val startScale = renderer.scale
                            val startX = renderer.x
                            val startY = renderer.y

                            val targetScale = if (renderer.scale == fitScale.value) {
                                doubleTapScale.value
                            } else {
                                fitScale.value
                            }
                            val diff = 1 / targetScale - 1 / startScale

                            val position = secondDown.position
                            val max_x = renderer.maxX(targetScale)
                            val max_y = renderer.maxY(targetScale)
                            var x = renderer.x + (position.x / renderer.width - 0.5f) * diff
                            var y = renderer.y + (position.y / renderer.height - 0.5f) * diff
                            x = x.coerceIn(-max_x, max_x)
                            y = y.coerceIn(-max_y, max_y)

                            val px = (x - startX) / diff
                            val py = (y - startY) / diff

                            renderer.animationJob?.cancel()
                            renderer.animationJob = scope.launch {
                                animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                                    renderer.scale = startScale + (targetScale - startScale) * value
                                    val diff = 1 / renderer.scale - 1 / startScale
                                    renderer.setPos(
                                        (startX + px * diff).orZero(),
                                        (startY + py * diff).orZero()
                                    )
                                }
                            }
                        } else {
                            // double tap drag
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)

                            val dragPointerId = secondDown.id

                            val originalScale = renderer.scale
                            val originalX = renderer.x
                            val originalY = renderer.y
                            var totalDeltaY = 0f

                            renderer.animationJob?.cancel()

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

                                    val px = secondDown.position.x / renderer.width - 0.5f
                                    val py = secondDown.position.y / renderer.height - 0.5f

                                    val new_scale =
                                        originalScale * 10f.pow(2 * totalDeltaY / renderer.height)

                                    renderer.scale = new_scale
                                    val diff = 1 / renderer.scale - 1 / originalScale

                                    renderer.setPos(
                                        (originalX + px * diff).orZero(),
                                        (originalY + py * diff).orZero()
                                    )

                                    change.consume()
                                }
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 200 && renderer.scale > fitScale.value && renderer.scale < renderer.maxScale) {
                                // fling zoom
                                renderer.animationJob = scope.launch {
                                    val animation = Animatable(0f)
                                    animation.snapTo(0f)
                                    animation.animateDecay(velocity.y, exponentialDecay()) {
                                        val px = secondDown.position.x / renderer.width - 0.5f
                                        val py = secondDown.position.y / renderer.height - 0.5f

                                        val new_scale =
                                            originalScale * 10f.pow(2 * (totalDeltaY + value) / renderer.height)

                                        renderer.scale =
                                            new_scale.coerceIn(fitScale.value, renderer.maxScale)
                                        val diff = 1 / renderer.scale - 1 / originalScale

                                        val x = (originalX + px * diff).orZero()
                                        val y = (originalY + py * diff).orZero()

                                        val max_x = renderer.maxX()
                                        val max_y = renderer.maxY()

                                        renderer.setPos(
                                            x.coerceIn(-max_x, max_x),
                                            y.coerceIn(-max_y, max_y)
                                        )
                                    }
                                }
                            } else {
                                renderer.reset(
                                    scope,
                                    Offset(
                                        secondDown.position.x / renderer.width,
                                        secondDown.position.y / renderer.height
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

                        renderer.animationJob?.cancel()

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
                                    }
                                    velocityTracker.addPointerInputChange(change)
                                    single = false

                                    scaleOrigin = Offset(
                                        centroid.x / renderer.width.toFloat(),
                                        centroid.y / renderer.height.toFloat()
                                    )
                                } else if (single) {
                                    velocityTracker.addPointerInputChange(change)
                                }

                                val zoom = event.calculateZoom()
                                val pan = event.calculatePan()
                                acc += pan

                                if (acc.getDistance() > touchSlop && renderer.scale > fitScale.value) {
                                    view.parent?.requestDisallowInterceptTouchEvent(false)
                                }

                                if (zoom != 1f || pan != Offset.Zero) {
                                    val new_scale = renderer.scale * zoom
                                    val diff = 1 / new_scale - 1 / renderer.scale

                                    var x = renderer.x + (pan.x / renderer.width) / renderer.scale
                                    var y = renderer.y + (pan.y / renderer.height) / renderer.scale

                                    x += (centroid.x / renderer.width - 0.5f) * diff
                                    y += (centroid.y / renderer.height - 0.5f) * diff

                                    val max_x = renderer.maxX(new_scale)
                                    val max_y = renderer.maxY(new_scale)

                                    if (renderer.scale != new_scale ||
                                        (!view.scrollable(true) && !view.scrollable(false)) ||
                                        (view.scrollable(true) && x.coerceIn(
                                            -max_x,
                                            max_x
                                        ) != renderer.x) ||
                                        (view.scrollable(false) && y.coerceIn(
                                            -max_y,
                                            max_y
                                        ) != renderer.y)
                                    ) {
                                        renderer.scale = new_scale

                                        if (single) {
                                            x = x.coerceIn(-max_x, max_x)
                                            y = y.coerceIn(-max_y, max_y)
                                        }

                                        renderer.setPos(x.orZero(), y.orZero())

                                        view.parent?.requestDisallowInterceptTouchEvent(true)
                                        event.changes.forEach {
                                            if (it.positionChanged()) {
                                                it.consume()
                                            }
                                        }
                                    } else {
                                        view.parent?.requestDisallowInterceptTouchEvent(false)
                                    }
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        val max_x = renderer.maxX()
                        val max_y = renderer.maxY()

                        val velocity = velocityTracker.calculateVelocity()
                        if (
                            (renderer.scale >= fitScale.value) &&
                            (renderer.scale <= renderer.maxScale) &&
                            (lastEventTime - lastMoveTime) < 100 &&
                            (abs(velocity.x) > 400 || abs(velocity.y) > 400) &&
                            (renderer.x.coerceIn(-max_x, max_x) == renderer.x
                                    || renderer.y.coerceIn(-max_y, max_y) == renderer.y)
                        ) {
                            // fling pan
                            renderer.animationJob = scope.launch {
                                fling.snapTo(Offset.Zero)
                                var lastOffset = Offset.Zero
                                fling.animateDecay(
                                    Offset(velocity.x, velocity.y),
                                    exponentialDecay()
                                ) {
                                    val delta = value - lastOffset
                                    lastOffset = value
                                    val dx = (delta.x / renderer.width) / renderer.scale
                                    val dy = (delta.y / renderer.height) / renderer.scale
                                    renderer.setPos(
                                        (renderer.x + dx).coerceIn(-max_x, max_x).orZero(),
                                        (renderer.y + dy).coerceIn(-max_y, max_y).orZero()
                                    )
                                }
                            }
                        } else {
                            renderer.reset(scope, scaleOrigin)
                        }
                    }
                }
            },
        isOpaque = false
    ) {
        onSurface { surface, width, height ->
            try {
                renderer.init(surface, width, height)
                val image = Image(webgpu.device!!, bitmap)
                renderer.addImage(image)

                val ratiox = width.toFloat() / bitmap.width.toFloat()
                val ratioy = height.toFloat() / bitmap.height.toFloat()

                renderer.minScale = max(0.01f, min(ratiox, ratioy))
                renderer.maxScale = maxScale

                fitScale.value = if (startFitWidth && !startFitHeight) {
                    ratiox
                } else if (!startFitWidth && startFitHeight) {
                    ratioy
                } else if (startFitWidth) {
                    max(0.01f, min(ratiox, ratioy))
                } else {
                    1f
                }

                if (!startFitWidth && !startFitHeight) {
                    renderer.scale = 1f
                } else {
                    renderer.scale = fitScale.value
                }

                if (doubleTapScale.value < renderer.minScale) {
                    doubleTapScale.value = renderer.minScale
                }

                if (doubleTapScale.value <= fitScale.value) {
                    doubleTapScale.value = renderer.minScale * 1.5f
                }

                if (renderer.maxScale < fitScale.value) {
                    renderer.maxScale = fitScale.value * 2
                }

                if (zoomWide && renderer.imageWidth > renderer.imageHeight) {
                    renderer.scale = ratioy
                }
                val maxX = renderer.maxX()
                val maxY = renderer.maxY()

                renderer.x = -zoomStartPosition.x * maxX
                renderer.y = -zoomStartPosition.x * maxY

                renderer.render()

                awaitCancellation()
            } catch (e: Exception) {
                throw e
            } finally {
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

private suspend fun AwaitPointerEventScope.waitForCleanUp(
    pointerId: PointerId,
    timeout: Long,
    touchSlop: Float
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
} catch (e: TimeoutCancellationException) {
    null
} as PointerEvent?

private suspend fun AwaitPointerEventScope.waitForDown(timeout: Long) =
    try {
        withTimeout(timeout) {
            var down = awaitPointerEvent().changes.firstOrNull { it.pressed }
            while (down == null) {
                down = awaitPointerEvent().changes.firstOrNull { it.pressed }
            }
            down
        }
    } catch (e: TimeoutCancellationException) {
        null
    }

fun Float.orZero(): Float = if (this.isNaN()) 0f else this
