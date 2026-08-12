package ca.mpreg.webgpuviewer.viewer

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.time.Duration.Companion.milliseconds

open class ImagePage(var image: Image?) {
    class Dummy(override val width: Int, override val height: Int) : ImagePage(null)

    class Draw private constructor(image: Image) : ImagePage(image) {
        companion object {
            suspend operator fun invoke(width: Int, height: Int): Draw {
                return Draw(Image(width, height))
            }
        }

        val texture: GPUTexture? get() = image?.mipmaps?.firstOrNull()?.textures?.firstOrNull()
    }

    val isDecoded: Boolean get() = this !is Dummy && this !is Draw

    companion object {
        suspend operator fun invoke(
            pixels: ByteBuffer, width: Int, height: Int, createMipMaps: Boolean = true
        ): ImagePage {
            return ImagePage(Image(pixels, width, height, createMipMaps))
        }

        suspend operator fun invoke(bitmap: Bitmap, createMipMaps: Boolean = true): ImagePage {
            val buf = ByteBuffer.allocateDirect(bitmap.byteCount)
            bitmap.copyPixelsToBuffer(buf)
            return ImagePage(buf, bitmap.width, bitmap.height, createMipMaps)
        }

        suspend operator fun invoke(width: Int, height: Int): ImagePage {
            return Draw(width, height)
        }
    }

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    var animationJob: Job? = null
    var animationLoop: Job? = null
    var pages: List<Pair<Image, Int>>? = null

    var destroyed = false

    fun cleanup() {
        destroyed = true
        animationLoop?.cancel()
        animationLoop = null
        animationJob?.cancel()
        animationJob = null

        // Capture references before nulling
        val imageToCleanup = image
        val pagesToCleanup = pages
        image = null
        pages = null

        if (imageToCleanup != null || pagesToCleanup != null) {
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    WebGpuRenderer.withContext {
                        // If pages exist, they contain all frames including the first one (which is also image)
                        // So only clean pages OR image, not both, to avoid double cleanup
                        if (pagesToCleanup != null) {
                            pagesToCleanup.forEach { it.first.cleanup() }
                        } else {
                            imageToCleanup?.cleanup()
                        }
                    }
                } catch (e: Exception) {
                    Log.e("ImagePage", "Error during cleanup", e)
                }
            }
        }
    }

    fun startAnimationLoop(pages: List<Pair<Image, Int>>, invalidate: () -> Unit) {
        animationLoop?.cancel()
        this.pages = pages
        animationLoop = CoroutineScope(Dispatchers.Default).launch {
            var frame = 0
            while (true) {
                this@ImagePage.pages?.get(frame)?.let { image ->
                    this@ImagePage.image = image.first
                    invalidate()
                    delay(image.second.coerceAtLeast(0).milliseconds)
                }
                frame = (frame + 1) % (this@ImagePage.pages?.size ?: 1)
            }
        }
    }

    open val width get() = image?.width ?: 0
    open val height get() = image?.height ?: 0

    var trim: Rect? = null

    private val contentWidth: Float
        get() = parent?.width?.toFloat() ?: 0f

    private val contentHeight: Float
        get() {
            val parent = parent ?: return 0f
            return if (parent.avoidCutout && parent.cutoutTopPx > 0f) {
                parent.height - parent.cutoutTopPx
            } else {
                parent.height.toFloat()
            }
        }

    val homeScale: Float
        get() {
            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f
            val imageWidth = (trim?.width() ?: width).toFloat()
            val imageHeight = (trim?.height() ?: height).toFloat()
            return minOf(
                contentWidth / imageWidth, contentHeight / imageHeight
            ).coerceAtLeast(0.01f)
        }

    val homeX: Float
        get() {
            val trim = trim ?: return 0f
            val parent = parent ?: return 0f
            val center = (trim.left + trim.right) / 2f
            val maxX = maxX(homeScale)
            return ((0.5f * width - center) / parent.width).fastCoerceIn(-maxX, maxX)
        }

    val homeY: Float
        get() {
            val parent = parent ?: return 0f

            // trimBaseY centers the trim on screen
            val trimBaseY = trim?.let {
                (0.5f * height - (it.top + it.bottom) / 2f) / parent.height
            } ?: 0f

            val cutoutPx = parent.cutoutTopPx
            if (cutoutPx > 0f && parent.height > 0) {
                val visibleHeight = (trim?.height() ?: height).toFloat()
                val trimHeightOnScreen = visibleHeight * homeScale
                val trimTopY = (parent.height - trimHeightOnScreen) / 2f

                val cutoutOffset = when {
                    parent.alwaysAvoidCutout -> cutoutPx / 2f
                    trimTopY < cutoutPx -> cutoutPx - trimTopY
                    else -> 0f
                }

                if (cutoutOffset > 0f) {
                    return trimBaseY + cutoutOffset / parent.height / homeScale
                }
            }

            val halfRange = max(0f, (height.toFloat() / parent.height - 1 / homeScale) / 2)
            return trimBaseY.fastCoerceIn(-halfRange, halfRange)
        }

    val atHome: Boolean
        get() {
            val eps = 0.0001f
            return abs(x - homeX) < eps && abs(y - homeY) < eps && abs(scale - homeScale) < eps
        }

    var onInvalidate: (() -> Unit)? = null

    var parent: ImageViewerState? = null

    var minScale = -1f
        get() {
            if (field > 0) return field
            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f
            return minOf(contentWidth / width, contentHeight / height).coerceAtLeast(0.01f)
        }

    var dpi = Resources.getSystem().displayMetrics.densityDpi / 100f

    val doubleTapScale get() = max(dpi, minScale * 2)

    var maxScale = -1f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

    fun maxX(scale: Float): Float {
        val parent = parent ?: return 0f
        return max(0f, (width.toFloat() / parent.width - 1 / scale) / 2)
    }

    fun minY(scale: Float): Float {
        if (scale == homeScale) return homeY
        val parent = parent ?: return 0f
        return -max(0f, (height.toFloat() / parent.height - 1 / scale) / 2)
    }

    fun maxY(scale: Float): Float {
        if (scale == homeScale) return homeY
        val parent = parent ?: return 0f
        return max(0f, (height.toFloat() / parent.height - 1 / scale) / 2)
    }

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

        @Suppress("NAME_SHADOWING") val targetScale = targetScale.fastCoerceIn(minScale, maxScale)

        val maxX = maxX(targetScale)
        val minY = minY(targetScale)
        val maxY = maxY(targetScale)

        val scaleChanging = targetScale != startScale
        val diffEnd = if (scaleChanging) 1 / targetScale - 1 / startScale else 1f

        val endX: Float
        val endY: Float

        if (origin != null && scaleChanging) {
            endX = (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(-maxX, maxX)
            endY = (startY + (origin.y - 0.5f) * diffEnd).fastCoerceIn(minY, maxY)
        } else if (origin != null) {
            endX = x.fastCoerceIn(-maxX, maxX)
            endY = y.fastCoerceIn(minY, maxY)
        } else {
            endX = targetX
            endY = targetY
        }

        animationJob = scope?.launch {
            animate(
                0f, 1f, animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
            ) { value, _ ->
                val currentScale = startScale + (targetScale - startScale) * value
                val c = if (scaleChanging) {
                    ((1 / currentScale - 1 / startScale) / diffEnd).fastCoerceIn(0f, 1f)
                } else {
                    value
                }

                setPos(
                    (startX + (endX - startX) * c).orZero(),
                    (startY + (endY - startY) * c).orZero(),
                    currentScale
                )
            }
        }
    }
}
