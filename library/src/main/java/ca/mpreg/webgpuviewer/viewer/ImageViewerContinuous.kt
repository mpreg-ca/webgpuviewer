package ca.mpreg.webgpuviewer.viewer

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
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.input.pointer.util.VelocityTracker
import androidx.compose.ui.input.pointer.util.addPointerInputChange
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.util.fastCoerceIn
import ca.mpreg.webgpuviewer.waitForCleanUp
import ca.mpreg.webgpuviewer.waitForDown
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.time.Duration.Companion.milliseconds

@Composable
fun ImageViewerContinuous(
    modifier: Modifier = Modifier,
    state: ImageViewerContinuousState,
) {
    val scope = rememberCoroutineScope()
    val view = LocalView.current
    val fling = remember { Animatable(Offset.Zero, Offset.VectorConverter) }

    val dpi = view.resources.displayMetrics.densityDpi / 100f
    val minScale = 1f
    val doubleTapScale = max(dpi, minScale * 2f)
    val maxScale = max(doubleTapScale * 2f, 4f)

    AndroidExternalSurface(
        modifier = modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                val doubleTapTimeout = viewConfiguration.doubleTapTimeoutMillis
                val touchSlop = viewConfiguration.touchSlop

                awaitEachGesture {
                    val firstDown = awaitFirstDown(pass = PointerEventPass.Initial)
                    state.animationJob?.cancel()
                    view.parent?.requestDisallowInterceptTouchEvent(true)

                    if (waitForCleanUp(firstDown.id, doubleTapTimeout, touchSlop) != null) {
                        // Tap - wait for possible double tap
                        val secondDown = waitForDown(doubleTapTimeout)
                        if (secondDown == null) {
                            // Single tap
                            state.onTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )
                            return@awaitEachGesture
                        }
                        if (waitForCleanUp(secondDown.id, doubleTapTimeout, touchSlop) != null) {
                            // Double tap: toggle zoom
                            if (state.scale > minScale + 0.1f) {
                                // Zoom out
                                state.animationJob = scope.launch {
                                    val startScale = state.scale
                                    val startOffsetX = state.offsetX
                                    animate(0f, 1f, animationSpec = tween(300)) { t, _ ->
                                        state.scale = startScale + (minScale - startScale) * t
                                        state.offsetX = startOffsetX * (1f - t)
                                        state.invalidate()
                                    }
                                }
                            } else {
                                // Zoom in at tap point
                                val px = secondDown.position.x / state.width - 0.5f
                                val py = secondDown.position.y / state.height - 0.5f
                                state.animationJob = scope.launch {
                                    val startScale = state.scale
                                    animate(0f, 1f, animationSpec = tween(300)) { t, _ ->
                                        val newScale =
                                            startScale + (doubleTapScale - startScale) * t
                                        val scrollDelta =
                                            py * (newScale - state.scale) * state.height / newScale
                                        state.offsetX = -px * (1f - minScale / newScale)
                                        state.scale = newScale
                                        state.scrollBy(scrollDelta)
                                        state.invalidate()
                                    }
                                }
                            }
                        } else {
                            // Double tap drag: zoom by dragging
                            val velocityTracker = VelocityTracker()
                            velocityTracker.addPointerInputChange(secondDown)
                            val dragPointerId = secondDown.id
                            val originalScale = state.scale
                            val originalOffsetX = state.offsetX
                            val originalScrollY = state.scrollY
                            val px = secondDown.position.x / state.width - 0.5f
                            val py = secondDown.position.y / state.height - 0.5f
                            var totalDeltaY = 0f

                            while (true) {
                                val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                val change = event.changes.firstOrNull { it.id == dragPointerId }
                                if (change == null || change.changedToUp() || change.isConsumed) break

                                velocityTracker.addPointerInputChange(change)

                                if (change.positionChanged()) {
                                    val pan = event.calculatePan()
                                    totalDeltaY += pan.y

                                    if (totalDeltaY != 0f) {
                                        val newScale =
                                            originalScale * 10f.pow(2 * totalDeltaY / state.height)
                                        val diff = 1f / newScale - 1f / originalScale
                                        state.scale = newScale
                                        state.offsetX = originalOffsetX + px * diff
                                        state.scrollY = originalScrollY - py * diff * state.height
                                        state.invalidate()
                                        change.consume()
                                    }
                                }
                            }

                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 200 && state.scale in minScale..maxScale) {
                                // Fling zoom
                                state.animationJob = scope.launch {
                                    val animation = Animatable(0f)
                                    animation.animateDecay(velocity.y, exponentialDecay()) {
                                        val newScale =
                                            (originalScale * 10f.pow(2 * (totalDeltaY + value) / state.height)).fastCoerceIn(
                                                minScale, maxScale
                                            )
                                        if (newScale != state.scale) {
                                            val diff = 1f / newScale - 1f / originalScale
                                            state.scale = newScale
                                            state.offsetX = originalOffsetX + px * diff
                                            state.scrollY =
                                                originalScrollY - py * diff * state.height
                                            state.invalidate()
                                        }
                                    }
                                }
                            } else {
                                // Snap scale and offsetX back if overshot
                                val targetScale = state.scale.fastCoerceIn(minScale, maxScale)
                                val targetMaxOffsetX =
                                    max(0f, (targetScale - 1f) / (2f * targetScale))
                                val targetOffsetX =
                                    state.offsetX.fastCoerceIn(-targetMaxOffsetX, targetMaxOffsetX)
                                if (targetScale != state.scale || targetOffsetX != state.offsetX) {
                                    state.animationJob = scope.launch {
                                        val startScale = state.scale
                                        val startOffsetX = state.offsetX
                                        animate(0f, 1f, animationSpec = tween(300)) { t, _ ->
                                            state.scale =
                                                startScale + (targetScale - startScale) * t
                                            state.offsetX =
                                                startOffsetX + (targetOffsetX - startOffsetX) * t
                                            state.invalidate()
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Drag gesture
                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

                        var single = true
                        var lastZoomTime = firstDown.uptimeMillis
                        var zoomVelocity = 0f
                        var lastCentroid = Offset(0.5f, 0.5f)

                        var longPressed = false
                        val longPressJob = scope.launch {
                            delay(viewConfiguration.longPressTimeoutMillis.milliseconds)
                            longPressed = true
                            state.onLongTap?.invoke(
                                Offset(
                                    firstDown.position.x / state.width,
                                    firstDown.position.y / state.height
                                )
                            )
                        }

                        do {
                            val event = awaitPointerEvent()
                            val canceled = event.changes.any { it.isConsumed }
                            if (!canceled) {
                                val change = event.changes[0]

                                if (event.changes.size > 1 && event.changes.all { it.pressed }) {
                                    if (single) {
                                        velocityTracker.resetTracking()
                                    }
                                    single = false
                                }

                                velocityTracker.addPointerInputChange(change)

                                val pan = event.calculatePan()
                                val zoom = event.calculateZoom()

                                if (pan != Offset.Zero || zoom != 1f) {
                                    longPressJob.cancel()

                                    if (zoom != 1f) {
                                        velocityTracker.resetTracking()
                                        val centroid = event.calculateCentroid(useCurrent = true)
                                        lastCentroid = Offset(
                                            centroid.x / state.width, centroid.y / state.height
                                        )
                                        val newScale = state.scale * zoom
                                        val diff = 1f / newScale - 1f / state.scale
                                        val cx = lastCentroid.x - 0.5f
                                        val cy = lastCentroid.y - 0.5f
                                        state.offsetX += cx * diff
                                        state.scrollBy(-cy * diff * state.height)
                                        state.scale = newScale

                                        // Track zoom velocity in log-scale space
                                        val now = change.uptimeMillis
                                        val dt = (now - lastZoomTime).coerceAtLeast(1L)
                                        val logZoom = ln(zoom) / (dt / 1000f)
                                        zoomVelocity = zoomVelocity * 0.5f + logZoom * 0.5f
                                        lastZoomTime = now
                                    } else {
                                        zoomVelocity *= 0.8f
                                    }

                                    if (single) {
                                        val maxOffsetX =
                                            max(0f, (state.scale - 1f) / (2f * state.scale))
                                        state.offsetX =
                                            (state.offsetX + pan.x / state.width / state.scale).fastCoerceIn(
                                                -maxOffsetX, maxOffsetX
                                            )
                                    } else {
                                        state.offsetX += pan.x / state.width / state.scale
                                    }
                                    state.scrollBy(-pan.y / state.scale)
                                    state.invalidate()
                                    event.changes.forEach { if (it.positionChanged()) it.consume() }
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        longPressJob.cancel()
                        if (longPressed) return@awaitEachGesture

                        if (!single && abs(zoomVelocity) > 0.5f && state.scale in minScale..maxScale) {
                            // Fling zoom
                            val cx = lastCentroid.x - 0.5f
                            val cy = lastCentroid.y - 0.5f
                            val startScale = state.scale
                            val startOffsetX = state.offsetX
                            val startScrollY = state.scrollY
                            state.animationJob = scope.launch {
                                val animation = Animatable(0f)
                                animation.animateDecay(zoomVelocity, exponentialDecay()) {
                                    val newScale =
                                        (startScale * exp(value)).fastCoerceIn(minScale, maxScale)
                                    val diff = 1f / newScale - 1f / startScale
                                    val maxOffsetX = max(0f, (newScale - 1f) / (2f * newScale))
                                    state.scale = newScale
                                    state.offsetX = (startOffsetX + cx * diff).fastCoerceIn(
                                        -maxOffsetX, maxOffsetX
                                    )
                                    state.scrollY = startScrollY - cy * diff * state.height
                                    state.invalidate()
                                }
                            }
                        } else if (state.scale < minScale) {
                            // Snap scale back up
                            state.animationJob = scope.launch {
                                val startScale = state.scale
                                val startOffsetX = state.offsetX
                                animate(0f, 1f, animationSpec = tween(200)) { t, _ ->
                                    state.scale = startScale + (minScale - startScale) * t
                                    state.offsetX = startOffsetX * (1f - t)
                                    state.invalidate()
                                }
                            }
                        } else if (state.scale > maxScale) {
                            // Snap scale back down
                            state.animationJob = scope.launch {
                                val startScale = state.scale
                                val startOffsetX = state.offsetX
                                val targetMaxOffsetX = max(0f, (maxScale - 1f) / (2f * maxScale))
                                val targetOffsetX =
                                    startOffsetX.fastCoerceIn(-targetMaxOffsetX, targetMaxOffsetX)
                                animate(0f, 1f, animationSpec = tween(200)) { t, _ ->
                                    state.scale = startScale + (maxScale - startScale) * t
                                    state.offsetX =
                                        startOffsetX + (targetOffsetX - startOffsetX) * t
                                    state.invalidate()
                                }
                            }
                        } else {
                            // Scale in bounds: fling pan or snap offsetX
                            val velocity = velocityTracker.calculateVelocity()
                            if (abs(velocity.y) > 400 || abs(velocity.x) > 400) {
                                state.animationJob = scope.launch {
                                    fling.snapTo(Offset.Zero)
                                    var lastOffset = Offset.Zero
                                    fling.animateDecay(
                                        Offset(velocity.x, velocity.y), exponentialDecay()
                                    ) {
                                        val delta = value - lastOffset
                                        lastOffset = value
                                        val maxOffsetX =
                                            max(0f, (state.scale - 1f) / (2f * state.scale))
                                        state.offsetX =
                                            (state.offsetX + delta.x / state.width / state.scale).fastCoerceIn(
                                                -maxOffsetX, maxOffsetX
                                            )
                                        state.scrollBy(-delta.y / state.scale)
                                        state.invalidate()
                                    }
                                }
                            } else {
                                val maxOffsetX = max(0f, (state.scale - 1f) / (2f * state.scale))
                                val clampedX = state.offsetX.fastCoerceIn(-maxOffsetX, maxOffsetX)
                                if (clampedX != state.offsetX) {
                                    state.animationJob = scope.launch {
                                        val startX = state.offsetX
                                        animate(0f, 1f, animationSpec = tween(200)) { t, _ ->
                                            state.offsetX = startX + (clampedX - startX) * t
                                            state.invalidate()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }, isOpaque = false
    ) {
        onSurface { surface, width, height ->
            try {
                state.init(scope, surface, width, height)
                state.invalidate()
                state.collect()
            } finally {
                state.cleanup()
            }
        }
    }
}
