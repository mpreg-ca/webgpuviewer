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
                            var totalDeltaY = 0f

                            while (true) {
                                val event = awaitPointerEvent(pass = PointerEventPass.Initial)
                                val change =
                                    event.changes.firstOrNull { it.id == dragPointerId && it.positionChanged() }
                                if (change == null || change.changedToUp() || change.isConsumed) break

                                velocityTracker.addPointerInputChange(change)
                                val pan = event.calculatePan()
                                totalDeltaY += pan.y

                                if (totalDeltaY != 0f) {
                                    val newScale =
                                        (originalScale * 10f.pow(2 * totalDeltaY / state.height)).fastCoerceIn(
                                            minScale, maxScale
                                        )
                                    state.scale = newScale
                                    state.invalidate()
                                    change.consume()
                                }
                            }
                        }
                    } else {
                        // Drag gesture
                        val velocityTracker = VelocityTracker()
                        velocityTracker.addPointerInputChange(firstDown)

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
                                velocityTracker.addPointerInputChange(change)

                                val pan = event.calculatePan()
                                val zoom = event.calculateZoom()

                                if (pan != Offset.Zero || zoom != 1f) {
                                    longPressJob.cancel()

                                    // Zoom
                                    if (zoom != 1f) {
                                        val centroid = event.calculateCentroid(useCurrent = true)
                                        val newScale =
                                            (state.scale * zoom).fastCoerceIn(minScale, maxScale)
                                        val cy = centroid.y / state.height - 0.5f
                                        val zoomScrollDelta =
                                            cy * (newScale - state.scale) * state.height / newScale
                                        state.scale = newScale
                                        state.scrollBy(zoomScrollDelta)
                                    }

                                    // Pan
                                    val maxOffsetX =
                                        max(0f, (state.scale - 1f) / (2f * state.scale))
                                    state.offsetX =
                                        (state.offsetX + pan.x / state.width / state.scale).fastCoerceIn(
                                            -maxOffsetX, maxOffsetX
                                        )
                                    state.scrollBy(-pan.y / state.scale)
                                    state.invalidate()
                                    event.changes.forEach { if (it.positionChanged()) it.consume() }
                                }
                            }
                        } while (!canceled && event.changes.any { it.pressed })

                        longPressJob.cancel()
                        if (longPressed) return@awaitEachGesture

                        // Fling
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
                        }

                        // Snap scale back if below min
                        if (state.scale < minScale) {
                            state.animationJob = scope.launch {
                                animate(state.scale, minScale, animationSpec = tween(200)) { v, _ ->
                                    state.scale = v
                                    state.invalidate()
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
