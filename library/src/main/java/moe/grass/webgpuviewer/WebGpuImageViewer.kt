package moe.grass.webgpuviewer

import android.graphics.Bitmap
import android.util.Log
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
import kotlinx.coroutines.Job
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

@Composable
fun WebGpuImageViewer(
    bitmap: Bitmap,
    doubleTapScale: Float = LocalView.current.resources.displayMetrics.densityDpi / 200f,
    maxScale: Float = LocalView.current.resources.displayMetrics.densityDpi / 100f,
    zoomStartPosition: Offset = Offset(0f, 0f),
    startFitWidth: Boolean = false,
    startFitHeight: Boolean = false,
    modifier: Modifier = Modifier
) {
    val view = LocalView.current
    val renderer = remember { WebGpuRenderer() }
    val renderChannel = remember { Channel<Float>(Channel.CONFLATED) }
    val scope = rememberCoroutineScope()
    val animationJob = remember { mutableStateOf<Job?>(null) }
    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    var fitScale = 1f

    val reset: (origin: Offset) -> Unit = { origin ->
        animationJob.value?.cancel()
        val startScale = renderer.scale
        val startX = renderer.x
        val startY = renderer.y

        val targetScale = startScale.coerceIn(renderer.minScale, maxScale)
        val max_x = max(
            0f,
            (renderer.image_width.toFloat() / renderer.width - 1 / targetScale) / 2
        )
        val max_y = max(
            0f,
            (renderer.image_height.toFloat() / renderer.height - 1 / targetScale) / 2
        )

        var px: Float
        var py: Float

        if (targetScale != startScale) {
            val diff = 1 / targetScale - 1 / renderer.scale
            var x = renderer.x + (origin.x - 0.5f) * diff
            var y = renderer.y + (origin.y - 0.5f) * diff
            x = x.coerceIn(-max_x, max_x)
            y = y.coerceIn(-max_y, max_y)
            px = (x - startX) / diff
            py = (y - startY) / diff
        } else {
            px = (renderer.x.coerceIn(-max_x, max_x) - startX)
            py = (renderer.y.coerceIn(-max_y, max_y) - startY)
        }

        animationJob.value = scope.launch {
            animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                renderer.scale = startScale + (targetScale - startScale) * value
                val diff = if (renderer.scale != startScale) {
                    1 / renderer.scale - 1 / startScale
                } else {
                    value
                }
                renderer.x = startX + px * diff
                renderer.y = startY + py * diff
                renderChannel.trySend(0f)
            }
        }
    }

    AndroidExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                if (!renderer.ready) {
                    return@pointerInput
                }

                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    reset(Offset(0.5f, 0.5f))

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

                            var targetScale: Float
                            val px: Float
                            val py: Float

                            if (renderer.scale == fitScale) {
                                targetScale = max(doubleTapScale, fitScale)
                                val diff = 1 / targetScale - 1 / renderer.scale

                                val max_x = max(
                                    0f,
                                    (renderer.image_width.toFloat() / renderer.width - 1 / targetScale) / 2
                                )
                                val max_y = max(
                                    0f,
                                    (renderer.image_height.toFloat() / renderer.height - 1 / targetScale) / 2
                                )
                                var x =
                                    renderer.x + (secondDown.position.x / renderer.width - 0.5f) * diff
                                var y =
                                    renderer.y + (secondDown.position.y / renderer.height - 0.5f) * diff
                                x = x.coerceIn(-max_x, max_x)
                                y = y.coerceIn(-max_y, max_y)

                                px = (x - startX) / diff
                                py = (y - startY) / diff
                            } else {
                                targetScale = fitScale
                                val diff = 1 / targetScale - 1 / startScale
                                px = -startX / diff
                                py = -startY / diff
                            }

                            animationJob.value?.cancel()
                            animationJob.value = scope.launch {
                                animate(0f, 1f, animationSpec = tween(300)) { value, _ ->
                                    renderer.scale = startScale + (targetScale - startScale) * value
                                    val diff = 1 / renderer.scale - 1 / startScale
                                    renderer.x = startX + px * diff
                                    renderer.y = startY + py * diff
                                    renderChannel.trySend(0f)
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

                            animationJob.value?.cancel()

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

                                    renderer.x = originalX + px * diff
                                    renderer.y = originalY + py * diff
                                    renderChannel.trySend(0f)
                                    change.consume()
                                }
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 200 && renderer.scale > fitScale && renderer.scale < maxScale) {
                                // fling zoom
                                animationJob.value = scope.launch {
                                    val animation = Animatable(0f)
                                    animation.snapTo(0f)
                                    animation.animateDecay(velocity.y, exponentialDecay()) {
                                        val px = secondDown.position.x / renderer.width - 0.5f
                                        val py = secondDown.position.y / renderer.height - 0.5f

                                        val new_scale =
                                            originalScale * 10f.pow(2 * (totalDeltaY + value) / renderer.height)

                                        renderer.scale =
                                            new_scale.coerceIn(fitScale, maxScale)
                                        val diff = 1 / renderer.scale - 1 / originalScale

                                        val x = originalX + px * diff
                                        val y = originalY + py * diff

                                        val max_x = max(
                                            0f,
                                            (renderer.image_width.toFloat() / renderer.width - 1 / renderer.scale) / 2
                                        )
                                        val max_y = max(
                                            0f,
                                            (renderer.image_height.toFloat() / renderer.height - 1 / renderer.scale) / 2
                                        )

                                        renderer.x = x.coerceIn(-max_x, max_x)
                                        renderer.y = y.coerceIn(-max_y, max_y)
                                        renderChannel.trySend(0f)
                                    }
                                }
                            } else {
                                reset(
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

                        animationJob.value?.cancel()

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

                                if (acc.getDistance() > touchSlop && renderer.scale > fitScale) {
                                    view.parent?.requestDisallowInterceptTouchEvent(false)
                                }

                                if (zoom != 1f || pan != Offset.Zero) {
                                    val new_scale = renderer.scale * zoom
                                    val diff = 1 / new_scale - 1 / renderer.scale

                                    var x = renderer.x + (pan.x / renderer.width) / renderer.scale
                                    var y = renderer.y + (pan.y / renderer.height) / renderer.scale

                                    x += (centroid.x / renderer.width - 0.5f) * diff
                                    y += (centroid.y / renderer.height - 0.5f) * diff

                                    val max_x = max(
                                        0f,
                                        (renderer.image_width.toFloat() / renderer.width - 1 / new_scale) / 2
                                    )
                                    val max_y = max(
                                        0f,
                                        (renderer.image_height.toFloat() / renderer.height - 1 / new_scale) / 2
                                    )

                                    if (renderer.scale != new_scale ||
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
                                        renderer.x = x
                                        renderer.y = y
                                        view.parent?.requestDisallowInterceptTouchEvent(true)
                                        event.changes.forEach {
                                            if (it.positionChanged()) {
                                                it.consume()
                                            }
                                        }
                                    } else {
                                        view.parent?.requestDisallowInterceptTouchEvent(false)
                                    }

                                    renderChannel.trySend(0f)
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        val max_x = max(
                            0f,
                            (renderer.image_width.toFloat() / renderer.width - 1 / renderer.scale) / 2
                        )
                        val max_y = max(
                            0f,
                            (renderer.image_height.toFloat() / renderer.height - 1 / renderer.scale) / 2
                        )

                        val velocity = velocityTracker.calculateVelocity()
                        if (
                            (renderer.scale >= fitScale) &&
                            (renderer.scale <= maxScale) &&
                            (lastEventTime - lastMoveTime) < 100 &&
                            (abs(velocity.x) > 400 || abs(velocity.y) > 400) &&
                            (renderer.x.coerceIn(
                                -max_x,
                                max_x
                            ) == renderer.x || renderer.y.coerceIn(-max_y, max_y) == renderer.y)
                        ) {
                            Log.i("webgpurenderer", "fling pan")
                            // fling pan
                            animationJob.value = scope.launch {
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
                                    renderer.x = (renderer.x + dx).coerceIn(-max_x, max_x)
                                    renderer.y = (renderer.y + dy).coerceIn(-max_y, max_y)
                                    renderChannel.trySend(0f)
                                }
                            }
                        } else {
                            Log.i("webgpurenderer", "reset")
                            reset(scaleOrigin)
                        }
                    }
                }
            }
    ) {
        onSurface { surface, width, height ->
            try {
                renderer.init(bitmap, surface, width, height)
                fitScale = if (startFitWidth && !startFitHeight) {
                    renderer.ratiox
                } else if (startFitHeight && !startFitWidth) {
                    renderer.ratioy
                } else {
                    max(0.01f, min(renderer.ratiox, renderer.ratioy))
                }
                if (!startFitWidth && !startFitHeight) {
                    renderer.scale = 1f
                } else {
                    renderer.scale = fitScale
                }

                val max_x = max(
                    0f,
                    (renderer.image_width.toFloat() / renderer.width - 1 / renderer.scale) / 2
                )
                val max_y = max(
                    0f,
                    (renderer.image_height.toFloat() / renderer.height - 1 / renderer.scale) / 2
                )

                renderer.x = -zoomStartPosition.x * max_x
                renderer.y = -zoomStartPosition.y * max_y

                renderer.render()

                renderChannel.receiveAsFlow().collect {
                    renderer.render()
                }
            } finally {
                animationJob.value?.cancel()
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
