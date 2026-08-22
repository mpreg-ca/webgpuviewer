package ca.mpreg.webgpuviewer.viewer

import android.graphics.Bitmap
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
import kotlinx.coroutines.yield
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.time.Duration.Companion.milliseconds

/**
 * ImagePage holds one or two images with shared transform (x, y, scale).
 * 
 * For single page: images = [image]
 * For dual page spread: images = [leftImage, rightImage]
 * 
 * Handles:
 * - Transform state (x, y, scale) and home position calculations
 * - Animation (pan/zoom animations, animated image frame loops)
 * - Image lifecycle (cleanup)
 */
open class ImagePage(val images: List<Image?>) {

    constructor(image: Image?) : this(listOf(image))
    constructor(left: Image?, right: Image?) : this(listOf(left, right))

    /** Placeholder page with dimensions but no image data */
    class Dummy(override val width: Int, override val height: Int) : ImagePage(emptyList())

    /** Drawable page for rendering progress indicators */
    class Draw private constructor(image: Image) : ImagePage(image) {
        companion object {
            suspend operator fun invoke(width: Int, height: Int): Draw {
                return Draw(Image(width, height))
            }
        }

        init {
            // Content changes every frame while drawn into - not worth the filtered/tiled paths'
            // extra sharpness or the fast path's linear-light correction.
            highQuality = false
        }

        val texture: GPUTexture?
            get() = images.firstOrNull()?.mipmaps?.firstOrNull()?.textures?.firstOrNull()
    }

    companion object {
        /**
         * Shared scope for fire-and-forget GPU cleanup work.
         * Lives for the application lifetime; individual cleanups are tiny and non-cancellable anyway.
         */
        private val cleanupScope = CoroutineScope(Dispatchers.Default)

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

        suspend fun drawable(width: Int, height: Int): Draw = Draw(width, height)
    }

    val isDecoded: Boolean
        get() = this !is Dummy && this !is Draw && images.any { it != null }

    /**
     * True once [cleanup] has run and the images are gone or going.
     *
     * Volatile because it is set on whatever thread evicts the page but read on the GPU thread,
     * which uses it to skip drawing a page whose textures are being freed. A render snapshot is
     * captured on the main thread and drawn later, so it can outlive the page it names.
     */
    @Volatile
    var destroyed = false
        private set

    /** If true, this page owns its images and will clean them up. If false, images are borrowed. */
    var ownsImages: Boolean = true

    /**
     * When false, this page skips [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s tile cache
     * entirely and its fast path renders through [ca.mpreg.webgpuviewer.renderer.RenderPage.renderPlain]
     * instead of [ca.mpreg.webgpuviewer.renderer.RenderPage.renderFast] - for content not worth
     * either path's extra correctness or sharpness, such as [Draw]'s loading placeholder (always
     * false, see its `init`) or an app-drawn transition/error bitmap.
     */
    var highQuality: Boolean = true

    var scale: Float = 1f
    var x: Float = 0f
    var y: Float = 0f

    fun setPos(x: Float = this.x, y: Float = this.y, scale: Float = this.scale) {
        if (this.x == x && this.y == y && this.scale == scale) return
        this.x = x
        this.y = y
        this.scale = scale
        onInvalidate?.invoke()
    }

    /** Total width (sum of image widths) */
    open val width: Int
        get() = images.filterNotNull().sumOf { it.width }

    /** Total height (max of image heights) */
    open val height: Int
        get() = images.filterNotNull().maxOfOrNull { it.height } ?: 0

    /** 
     * Visible width after trim. For dual pages, inner edges are ignored.
     */
    val trimWidth: Int
        get() {
            if (images.size == 2) {
                val left = images[0]
                val right = images[1]
                // Left: from left trim to full width (ignore right trim)
                val leftW = left?.let { it.width - (it.trim?.left ?: 0) } ?: 0
                // Right: from 0 to right trim (ignore left trim)
                val rightW = right?.let { it.trim?.right ?: it.width } ?: 0
                return leftW + rightW
            }
            return images.sumOf { it?.trim?.width() ?: it?.width ?: 0 }
        }

    /** Visible height after trim (max of trim heights) */
    val trimHeight: Int
        get() = images.maxOfOrNull { it?.trim?.height() ?: it?.height ?: 0 } ?: 0

    var animationJob: Job? = null
    var animationTargetX: Float? = null
    var animationTargetY: Float? = null
    var animationTargetScale: Float? = null

    /**
     * True while [scale] is being animated - by [animateTo] or externally (e.g. fling-zoom decay,
     * which sets this directly). Gates the tile cache, which otherwise can't tell a settled scale
     * from a spring that's merely repeating a value for a frame mid-flight.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    var homeScaleOverride: Float? = null
    var homeXOverride: Float? = null
    var homeYOverride: Float? = null

    private var animationLoop: Job? = null
    private var frames: List<Pair<Image, Int>>? = null
    private var currentFrameImage: Image? = null

    /** True while an animation frame loop owns [image]. The tile cache skips animated pages. */
    val isAnimated: Boolean
        get() = frames != null

    /** Incremented each time the animation frame changes. Used by the render cache to detect stale frames. */
    @Volatile
    var frameVersion: Int = 0
        private set

    /** Current image for rendering (may change during animation) */
    val image: Image?
        get() = currentFrameImage ?: images.firstOrNull()

    fun startAnimationLoop(frames: List<Pair<Image, Int>>, invalidate: () -> Unit) {
        animationLoop?.cancel()
        this.frames = frames
        currentFrameImage = frames.firstOrNull()?.first

        // Use the page's scope if available, otherwise use the shared background scope
        val loopScope = scope ?: cleanupScope
        animationLoop = loopScope.launch {
            var frameIndex = 0
            while (true) {
                this@ImagePage.frames?.getOrNull(frameIndex)?.let { (img, duration) ->
                    currentFrameImage = img
                    frameVersion++
                    invalidate()
                    delay(duration.coerceAtLeast(0).milliseconds)
                } ?: break
                frameIndex = (frameIndex + 1) % (this@ImagePage.frames?.size ?: 1)
            }
        }
    }

    var parent: ImageViewerState? = null
        set(value) {
            val wasNull = field == null
            field = value
            // Initialize to home when parent first set
            if (wasNull && value != null && x == 0f && y == 0f && scale == 1f) {
                x = homeX
                y = homeY
                scale = homeScale
            }
        }

    var scope: CoroutineScope? = null
    var onInvalidate: (() -> Unit)? = null

    private val contentWidth: Float
        get() = parent?.width?.toFloat() ?: 0f

    private val contentHeight: Float
        get() {
            val parent = parent ?: return 0f
            return if (parent.avoidCutout && parent.cutoutTopPx > 0f) parent.height - parent.cutoutTopPx else parent.height.toFloat()
        }

    /** True if this page uses half-screen layout (dual page or single LEFT/RIGHT) */
    private val isHalfWidth: Boolean
        get() = images.size == 2 || images.firstOrNull()?.position.let {
            it == Image.Position.LEFT || it == Image.Position.RIGHT
        }

    val homeScale: Float
        get() {
            homeScaleOverride?.let { return it.coerceAtLeast(0.01f) }

            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen, no trim
                val halfWidth = contentWidth / 2f
                return images.filterNotNull().minOfOrNull { img ->
                    minOf(halfWidth / img.width, contentHeight / img.height)
                }?.coerceAtLeast(0.01f) ?: 0.01f
            }

            // Single SINGLE page: fit trim to full screen
            val w = trimWidth.toFloat().takeIf { it > 0f } ?: return 0.01f
            val h = trimHeight.toFloat().takeIf { it > 0f } ?: return 0.01f
            return minOf(contentWidth / w, contentHeight / h).coerceAtLeast(0.01f)
        }

    val homeX: Float
        get() {
            homeXOverride?.let { return it }
            val parent = parent ?: return 0f

            // Half-width layout: no trim offset
            if (isHalfWidth) return 0f

            // Single SINGLE page: center the trim
            if (images.all { it?.trim == null }) return 0f
            val trim = images.firstOrNull()?.trim ?: return 0f
            val center = (trim.left + trim.right) / 2f
            val maxX = maxX(homeScale)
            return ((0.5f * width - center) / parent.width).fastCoerceIn(-maxX, maxX)
        }

    val homeY: Float
        get() {
            homeYOverride?.let { return it }
            val parent = parent ?: return 0f

            // Half-width layout: no trim, just cutout
            if (isHalfWidth) {
                if (parent.cutoutTopPx > 0f && parent.height > 0) {
                    val imageOnScreen = height * homeScale
                    // homeScale fits to contentHeight; image is centered in full parent.height
                    val imageTopY = (parent.height - imageOnScreen) / 2f
                    val cutoutOffset = when {
                        parent.alwaysAvoidCutout -> parent.cutoutTopPx / 2f
                        imageTopY < parent.cutoutTopPx -> parent.cutoutTopPx - imageTopY
                        else -> 0f
                    }
                    if (cutoutOffset > 0f) {
                        return cutoutOffset / (homeScale * parent.height)
                    }
                }
                return 0f
            }

            // Single SINGLE page: use trim
            val trimTop =
                images.mapNotNull { img -> img?.let { it.trim?.top ?: 0 } }.minOrNull() ?: 0
            val trimBottom =
                images.mapNotNull { it?.trim?.bottom ?: it?.height }.maxOrNull() ?: height

            val trimBaseY = if (images.any { it?.trim != null }) {
                (0.5f * height - (trimTop + trimBottom) / 2f) / parent.height
            } else 0f

            if (parent.cutoutTopPx > 0f && parent.height > 0) {
                val imageOnScreen = height * homeScale
                // Image top in screen pixels when centered in full parent.height at y=0
                val imageTopY = (parent.height - imageOnScreen) / 2f
                // Trim top = image top + trimTop scaled to screen pixels
                val trimTopY = imageTopY + trimTop * homeScale
                val cutoutOffset = when {
                    parent.alwaysAvoidCutout -> parent.cutoutTopPx / 2f
                    trimTopY < parent.cutoutTopPx -> parent.cutoutTopPx - trimTopY
                    else -> 0f
                }
                if (cutoutOffset > 0f) {
                    return cutoutOffset / (homeScale * parent.height)
                }
            }

            // Clamp trimBaseY so the image edge doesn't go past the screen edge.
            // When image is smaller than screen, the blank margin on each side limits the shift.
            // When image is larger than screen, any trim centering is valid (full scroll range available).
            val halfRange = max(0f, (1 / homeScale - height.toFloat() / parent.height) / 2)
            return trimBaseY.fastCoerceIn(-halfRange, halfRange)
        }

    val atHome: Boolean
        get() {
            val eps = 0.0001f
            return abs(x - homeX) < eps && abs(y - homeY) < eps && atHomeScale
        }

    val atHomeScale: Boolean
        get() {
            val eps = 0.0001f
            return abs(scale - homeScale) < eps
        }

    var minScale = -1f
        get() {
            if (field > 0) return field
            if (contentWidth <= 0f || contentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen
                val halfWidth = contentWidth / 2f
                return images.filterNotNull().minOfOrNull { img ->
                    minOf(halfWidth / img.width, contentHeight / img.height)
                }?.coerceAtLeast(0.01f) ?: 0.01f
            }

            // Single SINGLE page
            return minOf(contentWidth / width, contentHeight / height).coerceAtLeast(0.01f)
        }

    var maxScale = -1f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

    val doubleTapScale: Float get() = max(minScale, homeScale) * 2

    fun maxX(scale: Float): Float {
        val parent = parent ?: return 0f
        return max(0f, (width.toFloat() / parent.width - 1 / scale) / 2)
    }

    fun minY(scale: Float): Float {
        if (abs(scale - homeScale) < 0.0001f) return homeY
        val parent = parent ?: return 0f
        return -max(0f, (height.toFloat() / parent.height - 1 / scale) / 2)
    }

    fun maxY(scale: Float): Float {
        if (abs(scale - homeScale) < 0.0001f) return homeY
        val parent = parent ?: return 0f
        return max(0f, (height.toFloat() / parent.height - 1 / scale) / 2)
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
        val minY = minY(targetScale)
        val maxY = maxY(targetScale)

        val scaleChanging = targetScale != startScale
        val diffEnd = if (scaleChanging) 1 / targetScale - 1 / startScale else 1f

        val endX = when {
            origin != null && scaleChanging -> (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(
                -maxX, maxX
            )

            origin != null -> x.fastCoerceIn(-maxX, maxX)
            else -> targetX
        }
        val endY = when {
            origin != null && scaleChanging -> (startY + (origin.y - 0.5f) * diffEnd).fastCoerceIn(
                minY, maxY
            )

            origin != null -> y.fastCoerceIn(minY, maxY)
            else -> targetY
        }

        animationJob = scope?.launch {
            animationTargetX = endX
            animationTargetY = endY
            animationTargetScale = targetScale
            if (scaleChanging) isScaleAnimating = true
            try {
                animate(
                    0f, 1f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                    )
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
            } finally {
                animationTargetX = null
                animationTargetY = null
                animationTargetScale = null
                isScaleAnimating = false
            }
        }
    }

    @Synchronized
    fun cleanup() {
        if (destroyed) return
        destroyed = true

        animationLoop?.cancel()
        animationLoop = null
        animationJob?.cancel()
        animationJob = null

        // Only clean images if we own them
        if (!ownsImages) {
            frames = null
            currentFrameImage = null
            return
        }

        val framesToClean = frames
        frames = null
        currentFrameImage = null

        // Frames include all images; otherwise clean individual images
        val imagesToClean = framesToClean?.map { it.first } ?: images.filterNotNull()

        if (imagesToClean.isNotEmpty()) {
            cleanupScope.launch {
                try {
                    // Eviction fires exactly when the viewer reaches a new page, so freeing a
                    // page's textures competes with the frames that are drawing the new one.
                    // Yield between images and stay off the render mutex, for the same reason
                    // uploads do. Dawn keeps a destroyed texture alive until the command buffers
                    // referencing it retire, so a frame already in flight is unaffected.
                    WebGpuRenderer.onDispatcher {
                        imagesToClean.forEach { image ->
                            image.cleanup()
                            yield()
                        }
                    }
                } catch (e: Exception) {
                    Log.e("ImagePage", "Cleanup error", e)
                }
            }
        }
    }

    /** Alias for cleanup() */
    fun destroy() = cleanup()
}
