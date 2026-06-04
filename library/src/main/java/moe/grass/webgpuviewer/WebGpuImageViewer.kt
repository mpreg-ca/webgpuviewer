package moe.grass.webgpuviewer

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
import androidx.compose.foundation.gestures.calculateCentroidSize
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow

@Composable
fun WebGpuImageViewer(
    bitmap: Bitmap,
    modifier: Modifier = Modifier
) {
    val view = LocalView.current
    val renderer = remember { WebGpuRenderer() }
    val renderChannel = remember { Channel<Float>(Channel.CONFLATED) }
    val scope = rememberCoroutineScope()
    val animationJob = remember { mutableStateOf<Job?>(null) }
    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    AndroidExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                animationJob.value?.cancel()

                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop)
                        ?: return@awaitEachGesture
                    val secondDown = waitForDown(doubleTapTimeout) ?: return@awaitEachGesture
                    val dragPointerId = secondDown.id

                    val originalScale = renderer.scale
                    val originalX = renderer.x
                    val originalY = renderer.y
                    var totalDeltaY = 0f

                    while (true) {
                        val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                        val dragChange =
                            event.changes.firstOrNull { it.id == dragPointerId && it.positionChanged() }

                        if (dragChange == null || dragChange.changedToUp() || dragChange.isConsumed) {
                            break
                        }

                        val pan = event.calculatePan()
                        if (pan != Offset.Zero) {
                            view.parent?.requestDisallowInterceptTouchEvent(true)
                            animationJob.value?.cancel()

                            val px = secondDown.position.x / renderer.width - 0.5f
                            val py = secondDown.position.y / renderer.height - 0.5f

                            totalDeltaY += pan.y
                            val new_scale =
                                originalScale * 10f.pow(2 * totalDeltaY / renderer.height)

                            renderer.scale = max(new_scale, renderer.min_scale)
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
                            dragChange.consume()
                        }
                    }
                }
            }
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop)
                        ?: return@awaitEachGesture
                    val secondDown = waitForDown(doubleTapTimeout) ?: return@awaitEachGesture
                    waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop)
                        ?: return@awaitEachGesture

                    val startScale = renderer.scale
                    val startX = renderer.x
                    val startY = renderer.y

                    var targetScale = 1f
                    var targetX = 0f
                    var targetY = 0f

                    animationJob.value?.cancel()

                    if (renderer.scale == renderer.min_scale) {
                        val new_scale = 1.5f
                        val diff = 1 / new_scale - 1 / renderer.scale

                        targetScale = max(new_scale, renderer.min_scale)

                        targetX =
                            renderer.x + (secondDown.position.x / renderer.width - 0.5f) * diff
                        targetY =
                            renderer.y + (secondDown.position.y / renderer.height - 0.5f) * diff
                    } else {
                        targetX = 0f
                        targetY = 0f
                        targetScale = renderer.min_scale
                    }

                    animationJob.value = scope.launch {
                        animate(0f, 1f, animationSpec = tween(300)) { value, _ ->

                            val max_x = max(
                                0f,
                                (renderer.image_width.toFloat() / renderer.width - 1 / renderer.scale) / 2
                            )
                            val max_y = max(
                                0f,
                                (renderer.image_height.toFloat() / renderer.height - 1 / renderer.scale) / 2
                            )

                            renderer.scale = startScale + (targetScale - startScale) * value
                            val x = startX + (targetX - startX) * value
                            val y = startY + (targetY - startY) * value
                            renderer.x = x.coerceIn(-max_x, max_x)
                            renderer.y = y.coerceIn(-max_y, max_y)
                            renderChannel.trySend(0f)
                        }
                    }
                }
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    var zoom = 1f
                    var pan = Offset.Zero
                    var pastTouchSlop = false
                    val touchSlop = viewConfiguration.touchSlop

                    var moved = false
                    var dragOnly = true

                    val velocityTracker = VelocityTracker()

                    val firstDown =
                        awaitFirstDown(requireUnconsumed = true, pass = PointerEventPass.Initial)
                    velocityTracker.addPointerInputChange(firstDown)
                    var lastMoveTime = firstDown.uptimeMillis
                    var lastEventTime: Long

                    animationJob.value?.cancel()

                    if (renderer.scale > renderer.min_scale) {
                        view.parent?.requestDisallowInterceptTouchEvent(true)
                    }

                    do {
                        val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                        val change = event.changes[0]
                        lastEventTime = change.uptimeMillis

                        if (event.changes.size > 1) {
                            view.parent?.requestDisallowInterceptTouchEvent(true)
                            dragOnly = false
                        }


                        if (dragOnly) {
                            if (change.positionChanged()) {
                                lastMoveTime = change.uptimeMillis
                            }
                            velocityTracker.addPointerInputChange(change)
                        }

                        val zoomChange = event.calculateZoom()
                        val panChange = event.calculatePan()

                        if (!pastTouchSlop) {
                            zoom *= zoomChange
                            pan += panChange

                            val centroidSize = event.calculateCentroidSize(useCurrent = false)
                            val zoomMotion = abs(1 - zoom) * centroidSize
                            val panMotion = pan.getDistance()

                            if (zoomMotion > touchSlop || panMotion > touchSlop) {
                                pastTouchSlop = true
                            }
                        }

                        if (moved) {
                            event.changes.forEach {
                                if (it.positionChanged()) {
                                    it.consume()
                                }
                            }
                            view.parent?.requestDisallowInterceptTouchEvent(true)
                        }

                        if (pastTouchSlop) {
                            val centroid = event.calculateCentroid(useCurrent = false)
                            if (zoomChange != 1f || panChange != Offset.Zero) {
                                val new_scale = renderer.scale * zoomChange
                                val diff = 1 / new_scale - 1 / renderer.scale

                                var x =
                                    renderer.x + (panChange.x / renderer.width) / renderer.scale
                                var y =
                                    renderer.y + (panChange.y / renderer.height) / renderer.scale

                                val scale = max(new_scale, renderer.min_scale)

                                x += (centroid.x / renderer.width - 0.5f) * diff
                                y += (centroid.y / renderer.height - 0.5f) * diff

                                val max_x = max(
                                    0f,
                                    (renderer.image_width.toFloat() / renderer.width - 1 / scale) / 2
                                )
                                val max_y = max(
                                    0f,
                                    (renderer.image_height.toFloat() / renderer.height - 1 / scale) / 2
                                )

                                x = x.coerceIn(-max_x, max_x)
                                y = y.coerceIn(-max_y, max_y)

                                if (renderer.scale != scale) {
                                    moved = true
                                }

                                if (renderer.x != x && view.scrollable(true)) {
                                    moved = true
                                }
                                if (renderer.y != y && view.scrollable(false)) {
                                    moved = true
                                }

                                renderer.scale = scale
                                renderer.x = x
                                renderer.y = y

                                renderChannel.trySend(0f)
                            }

                            if (!moved) {
                                view.parent?.requestDisallowInterceptTouchEvent(false)
                            }
                        }
                    } while (event.changes.any { it.pressed })

                    if (moved && dragOnly && (lastEventTime - lastMoveTime) < 100) {
                        val velocity = velocityTracker.calculateVelocity()
                        if (abs(velocity.x) > 400 || abs(velocity.y) > 400) {
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
                                    val max_x = max(
                                        0f,
                                        (renderer.image_width.toFloat() / renderer.width - 1 / renderer.scale) / 2
                                    )
                                    val max_y = max(
                                        0f,
                                        (renderer.image_height.toFloat() / renderer.height - 1 / renderer.scale) / 2
                                    )
                                    renderer.x = (renderer.x + dx).coerceIn(-max_x, max_x)
                                    renderer.y = (renderer.y + dy).coerceIn(-max_y, max_y)
                                    renderChannel.trySend(0f)
                                }
                            }
                        }
                    }
                }
            },
    ) {
        onSurface { surface, width, height ->
            try {
                withContext(Dispatchers.Default) {
                    renderer.init(bitmap, surface, width, height)
                }

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
