package ca.mpreg.webgpuviewer

import android.view.View
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.AwaitPointerEventScope
import androidx.compose.ui.input.pointer.PointerEvent
import androidx.compose.ui.input.pointer.PointerEventTimeoutCancellationException
import androidx.compose.ui.input.pointer.PointerId
import androidx.compose.ui.input.pointer.changedToUp

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
