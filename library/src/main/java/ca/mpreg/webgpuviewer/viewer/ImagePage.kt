package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.graphics.Rect
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.util.fastCoerceIn
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.renderer.Image
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlin.math.max

open class ImagePage(val image: Image? = null) {
    class Dummy(override val width: Int, override val height: Int) : ImagePage(null)

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var animationJob: Job? = null

    open val width get() = image?.width ?: 0
    open val height get() = image?.height ?: 0

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

    var parent: ImageViewerState? = null

    var minScale = -1f
        get() = if (field > 0) field else parent?.getMinScale(width, height) ?: 1f

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    val doubleTapScale get() = max(dpi, minScale * 2)

    var maxScale = -1f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

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
            animate(
                0f, 1f, animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
            ) { value, _ ->
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
