package ca.mpreg.webgpuviewer.viewer

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDepthStencilAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.closeTo
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.Font
import ca.mpreg.webgpuviewer.draw.TextAlign
import ca.mpreg.webgpuviewer.draw.circle
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.draw.text
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.yield
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.time.Duration.Companion.milliseconds

/**
 * A page in the viewer, with shared transform (x, y, scale), pan/zoom-to-fit bounds, and
 * animation.
 *
 * [ImageSingle] is the ordinary single-page case; [ImageSpread] composes two [ImageSingle] pages
 * side by side for a dual-page spread. [Dummy] is a placeholder with known dimensions but no content.
 * [Render]/[Progress] draw their own content via [renderWith] instead of blitting an image.
 *
 * Handles:
 * - Transform state (x, y, scale) and home position calculations
 * - Pan/zoom animation
 * - Lifecycle (cleanup)
 */
open class ImagePage {

    /** Placeholder page with dimensions but no image data */
    class Dummy(override val width: Int, override val height: Int) : ImagePage()

    /**
     * ImagePage whose content is supplied by an app-overridden [render] instead of a decoded
     * image - for rendering progress indicators or other app-drawn content with its own shader.
     * Called instead of blitting a texture wherever this page would otherwise be drawn into the
     * regular view or a [ca.mpreg.webgpuviewer.transition.Transition]'s cache texture - unlike
     * [ImageSingle], it has no [ImageSingle.highQuality] to opt into either path, so it always
     * draws this way.
     *
     * Opens one pass per [render] call (no stencil attachment - unlike [ImageSingle], nothing here
     * needs [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s masking), shared by every [rect]/
     * [circle] call inside it rather than each opening its own.
     *
     * [renderWith] clears that pass's destination first - right whenever this page owns the whole
     * thing (a rotated screen buffer, or a [ca.mpreg.webgpuviewer.transition.Transition]'s
     * per-page cache slot). [renderLoaded] is the one exception:
     * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] draws several pages into one
     * shared screen texture, so a Render page's destination there is that whole shared texture,
     * not a page-sized one of its own - clearing it would blank every other visible page too.
     *
     * Override [backgroundColor] to fill a background before [render] runs: when
     * [renderWith] owns the whole destination, that fill is just its clear color, so it covers
     * the full screen (letterboxing beyond this page's own footprint, same as a transition
     * blending toward it); [renderLoaded] instead fills just [fillPage]'s scoped rect, since it
     * can't clear the shared texture. Leave it null to paint nothing and rely entirely on [render].
     */
    open class Render(override val width: Int, override val height: Int) : ImagePage() {

        // Always has drawable content via render(), unlike Dummy - needed so
        // Transition.getCachedTexture doesn't skip this page as if it were undecoded.
        override val isDecoded: Boolean get() = true

        // Unlike the base ImagePage default (fixed "home" position, right for Dummy - it has
        // nothing pan-worthy anyway), a Render page's own render() gets its *live* pan/zoom
        // transform - the same x/y/scale gestures already drive on any ImagePage - so e.g. a
        // custom Render page's own drawn content can track a drag/pinch the same way an
        // ImageSingle page's would, rather than being stuck rendering at (0, 0, 1) forever.
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            renderWith(encoder, x, y, scale, dst)
            return false
        }

        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            renderWith(encoder, x, y, scale, tex)
        }

        // render() treats dst as entirely its own canvas (see the class doc) - so unlike ImageSingle,
        // whose real content only ever occupies part of dst, this page's rect within a flat
        // render of it IS the whole thing, just panned/zoomed by its own live x/y/scale the same
        // way renderWith/renderCacheSeed already thread through. Without this, TransitionFlipLeft/
        // TransitionFlipRight/TransitionSphere - which all bail out on a null pageRect rather than
        // treating it as screen-shaped - would just never draw a Render page at all.
        override fun pageRect(dst: GPUTexture): FloatArray = floatArrayOf(
            0.5f + scale * (x - 0.5f),
            0.5f + scale * (y - 0.5f),
            0.5f + scale * (x + 0.5f),
            0.5f + scale * (y + 0.5f),
        )

        // Set by renderWith right before calling render(), and only valid for the duration of
        // that call - rect/circle/text read it instead of taking a pass parameter, since there's
        // only ever one pass open per render() call (see the class doc).
        private lateinit var pass: GPURenderPassEncoder

        /** Draws this page's content. Use [rect]/[circle]/[text] to draw into the open pass. */
        open fun render(dst: GPUTexture, x: Float, y: Float, scale: Float) {}

        @Volatile
        private var _renderVersion: Int = 0

        /** Bumped by [invalidate], so a page turn sees the drawn content change. */
        override val frameVersion: Int
            get() = _renderVersion

        /**
         * As [ImagePage.invalidate], bumping [frameVersion] so a page turn in flight re-seeds its
         * cache slot too - content that moves on its own (a progress ring, a countdown) would
         * otherwise freeze for the length of the turn.
         */
        override fun invalidate() {
            _renderVersion++
            super.invalidate()
        }

        protected fun rect(x1: Float, y1: Float, x2: Float, y2: Float, color: Int) =
            Draw.rect(pass, x1, y1, x2, y2, color)

        /**
         * Fills this page's own [width] x [height] footprint with [color] - unlike [rect]'s raw
         * [dst]-relative `[0,1]` coordinates (which always cover the *entire* [dst], full stop),
         * this is sized and positioned from this page's own declared [width]/[height] plus the
         * same [x]/[y]/[scale] [render] itself received, rather than assuming this page fills the
         * whole of [dst] the way [renderWith]'s full-[dst] clear does. Matters once [dst] is
         * shared with other pages ([ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState])
         * instead of being this page's own - filling all of `[0,1]` there would blank every other
         * visible page too, and even a same-sized fill would come out the wrong aspect ratio
         * whenever [dst] (the real screen) isn't [width]x[height]'s own aspect.
         */
        protected fun fillPage(dst: GPUTexture, x: Float, y: Float, scale: Float, color: Int) {
            val halfWidthFrac = scale * width / (2f * dst.width)
            val halfHeightFrac = scale * height / (2f * dst.height)
            val cx = 0.5f + scale * x
            val cy = 0.5f + scale * y
            rect(
                cx - halfWidthFrac,
                cy - halfHeightFrac,
                cx + halfWidthFrac,
                cy + halfHeightFrac,
                color
            )
        }

        protected fun circle(cx: Float, cy: Float, radius: Float, color: Int) =
            Draw.circle(pass, cx, cy, radius, color)

        protected fun text(
            dst: GPUTexture,
            font: Font,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(pass, dst, font, text, x, y, size, color, align, maxWidth)

        protected fun text(
            dst: GPUTexture,
            context: Context,
            fontFamily: FontFamily,
            text: String,
            x: Float,
            y: Float,
            size: Float,
            color: Int,
            weight: FontWeight = FontWeight.Normal,
            style: FontStyle = FontStyle.Normal,
            align: TextAlign = TextAlign.Left,
            maxWidth: Float = Float.POSITIVE_INFINITY,
        ) = Draw.text(
            pass, dst, context, fontFamily, text, x, y, size, color, weight, style, align, maxWidth
        )

        final override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first - see
            // ImagePage.renderWith's default for why that's still needed for a page (like Dummy)
            // that doesn't override this at all. [dst] is this call's own - a rotated screen
            // buffer or a Transition's per-page cache slot - so nothing else on screen depends on
            // whatever was already there.
            openPassAndRender(encoder, x, y, scale, dst, clear = true)
        }

        /**
         * As [renderWith], but loads [dst] instead of clearing it -
         * [ca.mpreg.webgpuviewer.viewer.ImageViewerContinuousState] uses this instead, since there
         * [dst] is one screen texture shared by several visible pages at once, and clearing it
         * would blank every other one. [render] is responsible for painting over every pixel of
         * its own footprint here - see [Progress] painting a full background rect before its
         * circle - since anything it doesn't touch keeps whatever [dst] already had (a decoded
         * neighbour's pixels, or simply undefined content the first time a fresh texture is used
         * - [ImageViewerContinuousState] clears once up front to guard against that).
         */
        internal fun renderLoaded(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            openPassAndRender(encoder, x, y, scale, dst, clear = false)
        }

        private fun argbToGPUColor(color: Int): GPUColor {
            val r = ((color shr 16) and 0xFF) / 255.0
            val g = ((color shr 8) and 0xFF) / 255.0
            val b = (color and 0xFF) / 255.0
            val a = ((color ushr 24) and 0xFF) / 255.0
            return GPUColor(r, g, b, a)
        }

        // Background fill driven by backgroundColor() (null by default; override it to
        // opt in - see Progress/an app's own transition placeholder) instead of a dedicated
        // property, since that's the exact same hook every Transition already reads for this
        // page's letterbox colour - one override covers both without the two ever disagreeing.
        private fun openPassAndRender(
            encoder: GPUCommandEncoder,
            x: Float,
            y: Float,
            scale: Float,
            dst: GPUTexture,
            clear: Boolean
        ) {
            val clearValue =
                backgroundColor?.let { argbToGPUColor(it) } ?: GPUColor(0.0, 0.0, 0.0, 0.0)

            val openedPass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = dst.createView(),
                            loadOp = if (clear) LoadOp.Clear else LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = clearValue
                        )
                    )
                )
            )
            pass = openedPass
            try {
                // clear=true already painted the whole dst this colour via clearValue above - a
                // page-scoped fillPage on top would be redundant. clear=false (the shared-texture
                // continuous-mode pass) has no clear to fall back on, so paint the footprint here.
                if (!clear) {
                    backgroundColor?.let { fillPage(dst, x, y, scale, it) }
                }
                render(dst, x, y, scale)
            } finally {
                openedPass.end()
            }
        }
    }

    /**
     * A page backed by a single decoded image, read straight off [currentImage]. [ImageSpread]
     * extends this to compose two side by side, overriding each member that needs both; pass
     * setup, tile cache and background fade math are shared as-is.
     */
    open class ImageSingle(val image: Image?) : ImagePage() {

        /** Animated from the start - no separate [startAnimationLoop] call needed. */
        constructor(frames: List<Pair<Image, Int>>) : this(frames.firstOrNull()?.first) {
            startAnimationLoop(frames)
        }

        /** If true, this page owns its image and will clean it up. If false, it's borrowed - see
         *  [ImageSpread], which composes existing pages without taking ownership of their images. */
        var ownsImage: Boolean = true

        /**
         * When false, this page skips [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s tile cache
         * entirely and its fast path renders through [renderPage] with `linear = false` instead of
         * the default `linear = true` - for content not worth either path's extra correctness or
         * sharpness, such as an app-drawn transition/error bitmap. A `var`, not a `val`, so
         * [ImageSpread] can override it to fan a set-through to both sides instead of just holding
         * its own copy.
         */
        open var highQuality: Boolean = true

        override val width: Int
            get() = image?.width ?: 0
        override val height: Int
            get() = image?.height ?: 0

        override val trimWidth: Int
            get() = image?.let { it.trim?.width() ?: it.width } ?: 0
        override val trimHeight: Int
            get() = image?.let { it.trim?.height() ?: it.height } ?: 0

        override fun xEdges(trimmed: Boolean): Pair<Float, Float> {
            if (!trimmed) return 0f to width.toFloat()
            val trim = image?.trim ?: return 0f to width.toFloat()
            return trim.left.toFloat() to trim.right.toFloat()
        }

        override fun yEdges(trimmed: Boolean): Pair<Int, Int>? {
            if (!trimmed) return null
            val trim = image?.trim ?: return null
            return trim.top to trim.bottom
        }

        private var animationLoop: Job? = null
        private var frames: List<Pair<Image, Int>>? = null
        private var currentFrameImage: Image? = null

        /** True while an animation frame loop owns [currentImage]. The tile cache skips animated pages. */
        override val isAnimated: Boolean
            get() = frames != null

        @Volatile
        private var _frameVersion: Int = 0

        /** Incremented each time the animation frame changes. Used by the render cache to detect stale frames. */
        override val frameVersion: Int
            get() = _frameVersion

        /** Current image for rendering (may change during animation) */
        open val currentImage: Image?
            get() = currentFrameImage ?: image

        /**
         * Runs [action] for each image drawn right now, with its pixel offset from the page
         * anchor - [currentImage] at 0 here, both sides for an [ImageSpread]. For callers that
         * place images with their own math ([ca.mpreg.webgpuviewer.renderer.TileRenderer],
         * [ImageViewerContinuousState]) instead of [renderPage].
         */
        internal open fun forEachImage(action: (image: Image, offsetX: Float) -> Unit) {
            currentImage?.let { action(it, 0f) }
        }

        /** True once at least one of this page's images has been uploaded and can be drawn. */
        internal open val hasUploadedImage: Boolean
            get() = currentImage?.mipmaps?.isNotEmpty() == true

        override val isDecoded: Boolean
            get() = currentImage != null

        /**
         * Left/right extent from this page's own anchor, in raw pixels at scale 1 - symmetric
         * halves of [width] by default; [ImageSpread] overrides for its asymmetric seam-based
         * shape.
         */
        internal open fun horizontalExtent(): Pair<Float, Float> {
            val half = width / 2f
            return half to half
        }

        /** As [ImagePage.invalidate], over the same [frameVersion] the animation loop drives. */
        override fun invalidate() {
            _frameVersion++
            super.invalidate()
        }

        fun startAnimationLoop(frames: List<Pair<Image, Int>>) {
            animationLoop?.cancel()
            this.frames = frames
            currentFrameImage = frames.firstOrNull()?.first

            // Use the page's scope if available, otherwise use the shared background scope
            val loopScope = scope ?: cleanupScope
            animationLoop = loopScope.launch {
                var frameIndex = 0
                while (true) {
                    this@ImageSingle.frames?.getOrNull(frameIndex)?.let { (img, duration) ->
                        currentFrameImage = img
                        // Keeps running off screen - frames stay in step with their durations,
                        // and invalidate() asks for a redraw only while there is one to ask for.
                        invalidate()
                        delay(duration.coerceAtLeast(0).milliseconds)
                    } ?: break
                    frameIndex = (frameIndex + 1) % (this@ImageSingle.frames?.size ?: 1)
                }
            }
        }

        override fun renderWith(
            encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
        ) {
            // Clears and draws in the same pass, rather than a separate clear pass first. No
            // stencil attachment - this fallback never needs TileRenderer's masking, unlike the
            // overrides below that bypass it entirely.
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = dst.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                renderPage(pass, dst, x, y, scale, linear = false, masked = false)
            } finally {
                pass.end()
            }
        }

        /** Opens a `LoadOp.Clear` pass on [dst], with a stencil attachment for [TileRenderer]'s masking. */
        private fun beginLivePass(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ) = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                ), depthStencilAttachment = GPURenderPassDepthStencilAttachment(
                    view = tiles.stencilViewFor(dst),
                    stencilLoadOp = LoadOp.Clear,
                    stencilStoreOp = StoreOp.Discard,
                    stencilClearValue = 0,
                )
            )
        )

        /** Opens a `LoadOp.Clear` pass on [tex], no stencil attachment - a transition's cache is never masked. */
        private fun beginCachePass(encoder: GPUCommandEncoder, tex: GPUTexture) =
            encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = tex.createView(),
                            loadOp = LoadOp.Clear,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )

        /**
         * As [ImagePage.drawLive]. Animated frames always want the fast path regardless of
         * [highQuality] (never worth a tile cache that would just churn every frame); a
         * non-[highQuality], non-animated page falls back to the plain [renderWith] (via `super`);
         * everything else goes through the tile cache, backfilling with [renderPage] wherever it
         * isn't covered yet. Opens its own pass (with a stencil attachment, for [TileRenderer]'s
         * masking) rather than sharing one from the caller - see [Render] for why that split exists.
         */
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            if (isAnimated) {
                val pass = beginLivePass(encoder, dst, tiles)
                try {
                    renderBackground(pass, dst, 0f, 0f, 1f)
                    renderPage(pass, dst, 0f, 0f, 1f)
                } finally {
                    pass.end()
                }
                return false
            }

            if (!highQuality) return super.drawLive(encoder, dst, tiles)

            val pass = beginLivePass(encoder, dst, tiles)
            try {
                // Background always drawn live first (its fades are position-dependent, never
                // from a stale tile) so it stays underneath everything else. Tiles draw next,
                // marking every pixel they cover in the stencil buffer tiles.draw() writes to;
                // renderPage then only shades what's left uncovered instead of the whole
                // viewport, since tiles.draw() already produced the right pixel wherever it drew.
                renderBackground(pass, dst, 0f, 0f, 1f)
                val covered = tiles.draw(pass, this, dst, 0f, 0f, 1f)
                if (!covered) {
                    renderPage(pass, dst, 0f, 0f, 1f)
                }
                if (fade < 1f) fadeRect(dst)?.let { drawFade(pass, it[0], it[1], it[2], it[3]) }
                return covered
            } finally {
                pass.end()
            }
        }

        /**
         * As [drawLive], seeding a transition's cache slot instead of the screen - same
         * isAnimated/highQuality precedence, but never with a stencil attachment (a transition's
         * cache is never stencil-masked either way).
         */
        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            if (isAnimated) {
                val pass = beginCachePass(encoder, tex)
                try {
                    renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                    if (fade < 1f) {
                        fadeRect(tex)?.let { drawFade(pass, it[0], it[1], it[2], it[3], false) }
                    }
                } finally {
                    pass.end()
                }
                return
            }

            if (!highQuality) {
                super.renderCacheSeed(encoder, tex, tiles)
                return
            }

            val pass = beginCachePass(encoder, tex)
            try {
                renderPage(pass, tex, 0f, 0f, 1f, masked = false)
                tiles.blitAvailableTiles(pass, this, tex)
                // A fade re-seeds the cache every frame (frameVersion), which is what lets a
                // page fade in mid-turn at all.
                if (fade < 1f) {
                    fadeRect(tex)?.let { drawFade(pass, it[0], it[1], it[2], it[3], false) }
                }
            } finally {
                pass.end()
            }
        }

        override fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
            if (!highQuality || isAnimated) null else tiles.availableTileKeys(this, tex)

        override fun renderIntoCache(
            encoder: GPUCommandEncoder,
            tex: GPUTexture,
            tiles: TileRenderer,
            identityMatches: Boolean
        ) {
            if (!identityMatches) {
                renderCacheSeed(encoder, tex, tiles)
                return
            }
            val pass = encoder.beginRenderPass(
                GPURenderPassDescriptor(
                    colorAttachments = arrayOf(
                        GPURenderPassColorAttachment(
                            view = tex.createView(),
                            loadOp = LoadOp.Load,
                            storeOp = StoreOp.Store,
                            clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                        )
                    )
                )
            )
            try {
                tiles.blitAvailableTiles(pass, this, tex)
            } finally {
                pass.end()
            }
        }

        override fun pageRect(dst: GPUTexture): FloatArray? {
            val image = currentImage ?: return null
            if (image.mipmaps.isEmpty()) return null
            return image.placement(dst, x, y, scale)
        }

        override val backgroundColor: Int?
            get() = currentImage?.backgroundColor

        override fun drawBackgroundColumns(
            pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
        ) {
            val image = currentImage ?: return
            if (image.mipmaps.isEmpty()) return
            // One image, so its column is the whole width - see [backgroundSpansFullWidth].
            Draw.rect(pass, offsetX, offsetY, offsetX + 1f, offsetY + 1f, image.backgroundColor)
        }

        /**
         * Draw this page into [pass], one bilinear tap per pixel. [linear]/[masked] pick the same
         * 4 pipelines as the image-level [RenderPage.renderFast]. Background handling is derived
         * from the pair rather than its own parameter: masked+linear skips it (that caller already
         * drew it via [renderBackground]), masked-only folds it into the masked draw, unmasked
         * always draws a plain one alongside.
         */
        fun renderPage(
            pass: GPURenderPassEncoder,
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            linear: Boolean = true,
            masked: Boolean = true
        ) {
            val variant = RenderPage.variantFor(linear, masked)
            forEachPlacedImage(dst, x, y, scale) { image, rect, placeX, placeY, placeScale ->
                if (!linear || !masked) {
                    drawImageBackground(pass, image, rect, scale, maskedBackground = masked)
                }
                for (tile in image.prepareTilesForRender(dst, placeX, placeY, placeScale)) {
                    RenderPage.drawTile(pass, dst, tile, variant)
                }
            }
        }

        /**
         * Draw just this page's per-image background colour, skipping the image itself - for
         * ImageViewerState's masked path, which draws this first (so it stays underneath
         * [ca.mpreg.webgpuviewer.renderer.TileRenderer.draw] and [renderPage]), or as the whole
         * draw once [ca.mpreg.webgpuviewer.renderer.TileRenderer] already covers the image itself.
         * The background's alpha depends on live pan/scale, so it's drawn every frame regardless
         * of tile coverage. Uses the stencil-pass-compatible rect pipeline - see
         * [RenderPage.drawMaskedRect] - since it always runs inside that pass.
         */
        fun renderBackground(
            pass: GPURenderPassEncoder, dst: GPUTexture, x: Float, y: Float, scale: Float
        ) = forEachPlacedImage(dst, x, y, scale) { image, rect, _, _, _ ->
            drawImageBackground(pass, image, rect, scale, maskedBackground = true)
        }

        /**
         * Draws [image]'s fading background rect, for [renderPage]'s `drawBackground` branch.
         * Alpha fades with distance from home/min scale or the page's pan bounds, so it only shows
         * near the edges of the zoom/pan range where the image itself doesn't fill the viewport.
         */
        private fun drawImageBackground(
            pass: GPURenderPassEncoder,
            image: Image,
            rect: FloatArray,
            scale: Float,
            maskedBackground: Boolean,
        ) {
            val parent = parent
            val minScale = minScale
            val homeScale = homeScale
            val currentScale = this.scale * scale

            val fadeDistancePixels = 200f
            val imageSize = (width.coerceAtLeast(height)).toFloat()

            fun proximity(anchorScale: Float): Float {
                if (anchorScale <= 0f) return 0f
                val deltaPixels = abs(imageSize * (currentScale - anchorScale))
                return (1f - deltaPixels / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundProximity(value: Float, lo: Float, hi: Float, pixelsPerUnit: Float): Float {
                val overflow = when {
                    value < lo -> lo - value
                    value > hi -> value - hi
                    else -> return 1f
                }
                return (1f - overflow * pixelsPerUnit / fadeDistancePixels).coerceIn(0f, 1f)
            }

            fun boundsProximityAt(anchorScale: Float): Float {
                if (parent == null || anchorScale <= 0f) return 0f
                val minX = minX(anchorScale)
                val maxX = maxX(anchorScale)
                val minY = minY(anchorScale)
                val maxY = maxY(anchorScale)
                val pixelsPerUnitX = parent.width.toFloat() * anchorScale
                val pixelsPerUnitY = parent.height.toFloat() * anchorScale
                return min(
                    boundProximity(this@ImageSingle.x, minX, maxX, pixelsPerUnitX),
                    boundProximity(this@ImageSingle.y, minY, maxY, pixelsPerUnitY)
                )
            }

            val bgAlpha = if (parent != null) {
                if (currentScale > minScale) {
                    boundsProximityAt(currentScale)
                } else {
                    val homeProximity = min(proximity(homeScale), boundsProximityAt(homeScale))
                    val minProximity = min(proximity(minScale), boundsProximityAt(minScale))
                    max(homeProximity, minProximity)
                }
            } else {
                1f
            }

            val origA = (image.backgroundColor ushr 24) and 0xFF
            val a = (origA * bgAlpha).toInt()
            if (a <= 0) return

            val x1 = if (backgroundSpansFullWidth) 0f else rect[0]
            val x2 = if (backgroundSpansFullWidth) 1f else rect[2]
            val origR = (image.backgroundColor shr 16) and 0xFF
            val origG = (image.backgroundColor shr 8) and 0xFF
            val origB = image.backgroundColor and 0xFF
            val r = (origR * bgAlpha).toInt()
            val g = (origG * bgAlpha).toInt()
            val b = (origB * bgAlpha).toInt()
            val bgColor = (a shl 24) or (r shl 16) or (g shl 8) or b
            if (maskedBackground) {
                RenderPage.drawMaskedRect(pass, x1, 0f, x2, 1f, bgColor)
            } else {
                Draw.rect(pass, x1, 0f, x2, 1f, bgColor)
            }
        }

        /**
         * True when the background colour paints the whole viewport rather than just the image's
         * rect - always so with one image, which has no neighbouring column to bleed into.
         * [ImageSpread] narrows it when it has two.
         */
        internal open val backgroundSpansFullWidth: Boolean
            get() = true

        /** Walks this page's image(s) via [forEachImage], placing each for [action] to draw against. */
        private fun forEachPlacedImage(
            dst: GPUTexture,
            x: Float,
            y: Float,
            scale: Float,
            action: (image: Image, rect: FloatArray, placeX: Float, placeY: Float, placeScale: Float) -> Unit
        ) {
            // The snapshot is captured on the main thread and drawn later, so the page may have
            // been evicted since - its images' buffers are gone, and touching one throws.
            if (destroyed) return

            forEachImage { img, srcOffsetX ->
                if (img.mipmaps.isNotEmpty()) {
                    val placeX = this.x + x + srcOffsetX / dst.width
                    val placeY = this.y + y
                    val placeScale = this.scale * scale
                    val rect = img.placement(dst, placeX, placeY, placeScale)
                    action(img, rect, placeX, placeY, placeScale)
                }
            }
        }

        @Synchronized
        override fun cleanup() {
            if (destroyed) return
            super.cleanup()

            animationLoop?.cancel()
            animationLoop = null

            // Only clean the image if we own it
            if (!ownsImage) {
                frames = null
                currentFrameImage = null
                return
            }

            val framesToClean = frames
            frames = null
            currentFrameImage = null

            // Frames include the image; otherwise clean it directly
            val imagesToClean = framesToClean?.map { it.first } ?: listOfNotNull(image)

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
    }

    /**
     * Two pages drawn side by side, sharing one pan/zoom transform and one
     * [ca.mpreg.webgpuviewer.renderer.TileRenderer] grid, so the seam bakes into whichever tile
     * straddles it instead of meeting two independently-snapped layers. Either side may be null,
     * e.g. a cover with no partner.
     *
     * Composes existing pages rather than owning decoded images: drawing and animation delegate to
     * whichever side is live via [ImageSingle.currentImage]/[isAnimated], so either can be an
     * animated GIF independently. Never cleans up [left]/[right] - whoever built them owns that.
     *
     * A side may also be a [Render] page. It has no image to place, so it sits out [forEachImage]
     * and the tile grid and is drawn into its half afterwards - see [drawRenderSides]. Everything
     * that measures a side reads the page, not a decoded image, so it lays out like an image one.
     */
    class ImageSpread(val left: ImagePage?, val right: ImagePage?) : ImageSingle(null) {

        /** Either side as an [ImageSingle] - null for a [Render] side, which has no image. */
        private val leftSingle: ImageSingle?
            get() = left as? ImageSingle
        private val rightSingle: ImageSingle?
            get() = right as? ImageSingle

        /** Runs [action] for each present side, with its pixel offset from the seam. */
        private inline fun forEachSide(action: (side: ImagePage, offsetX: Float) -> Unit) {
            left?.let { action(it, -0.5f * it.width) }
            right?.let { action(it, 0.5f * it.width) }
        }

        override var highQuality: Boolean
            get() = (leftSingle?.highQuality ?: true) && (rightSingle?.highQuality ?: true)
            set(value) {
                leftSingle?.highQuality = value
                rightSingle?.highQuality = value
            }

        override val isAnimated: Boolean
            get() = left?.isAnimated == true || right?.isAnimated == true

        override val frameVersion: Int
            get() = (left?.frameVersion ?: 0) + (right?.frameVersion ?: 0)

        /** No image of its own - [left]/[right] hold them, and [forEachImage] walks both. */
        override val currentImage: Image?
            get() = null

        /**
         * Each side sits half its own width out from the seam (the page anchor). A [Render] side
         * has no image to place and paints itself instead - see [drawRenderSides].
         */
        override fun forEachImage(action: (image: Image, offsetX: Float) -> Unit) {
            leftSingle?.currentImage?.let { action(it, -0.5f * it.width) }
            rightSingle?.currentImage?.let { action(it, 0.5f * it.width) }
        }

        override val hasUploadedImage: Boolean
            get() = leftSingle?.hasUploadedImage == true || rightSingle?.hasUploadedImage == true

        override val isDecoded: Boolean
            get() = left?.isDecoded == true || right?.isDecoded == true

        /** Two columns meeting at the seam, so neither may paint over the other half. */
        override val backgroundSpansFullWidth: Boolean
            get() = left == null || right == null

        /** As [ImageSingle.drawLive], then each [Render] side on top - see [drawRenderSides]. */
        override fun drawLive(
            encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
        ): Boolean {
            val covered = super.drawLive(encoder, dst, tiles)
            drawRenderSides(encoder, dst)
            return covered
        }

        override fun renderCacheSeed(
            encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
        ) {
            super.renderCacheSeed(encoder, tex, tiles)
            drawRenderSides(encoder, tex)
        }

        /** [frameVersion] is the sides' sum, so this page's own has nothing to bump. */
        override fun invalidate() {
            left?.invalidate()
            right?.invalidate()
            if (isOnScreen) onInvalidate?.invoke()
        }

        override fun covers(other: ImagePage): Boolean =
            this === other || left === other || right === other

        override fun attach(
            parent: ImageViewerState, scope: CoroutineScope?, onInvalidate: () -> Unit
        ) {
            super.attach(parent, scope, onInvalidate)
            left?.attach(parent, scope, onInvalidate)
            right?.attach(parent, scope, onInvalidate)
        }

        /** True when either side paints itself rather than blitting a decoded image. */
        private val hasRenderSide: Boolean
            get() = left is Render || right is Render

        /**
         * A [Render] side repaints every frame, so [ImageSingle]'s incremental tile blit would
         * leave it stale for the whole transition. Reseed the slot outright instead.
         */
        override fun renderIntoCache(
            encoder: GPUCommandEncoder,
            tex: GPUTexture,
            tiles: TileRenderer,
            identityMatches: Boolean
        ) = super.renderIntoCache(encoder, tex, tiles, identityMatches && !hasRenderSide)

        /**
         * Draws each [Render] side into its own half of [dst], after the image sides.
         *
         * Needs a pass of its own ([Render.renderLoaded] opens one), since a pass cannot nest
         * inside the one the image sides just drew through. It loads rather than clears: [dst]
         * already holds the other side by now.
         *
         * Placed with the spread transform shifted by that side's seam offset, as [forEachImage]
         * places an image side, so [Render.render]/[Render.fillPage] land in the right half
         * instead of treating [dst] as a page of their own.
         */
        private fun drawRenderSides(encoder: GPUCommandEncoder, dst: GPUTexture) {
            // As forEachPlacedImage: the page can have been evicted since the snapshot was taken.
            if (destroyed) return
            forEachSide { side, offsetX ->
                if (side is Render) {
                    side.renderLoaded(encoder, x + offsetX / dst.width, y, scale, dst)
                }
            }
        }

        override fun pageRect(dst: GPUTexture): FloatArray? {
            forEachSide { side, offsetX ->
                val image = (side as? ImageSingle)?.currentImage
                if (image != null && image.mipmaps.isNotEmpty()) {
                    return image.placement(dst, x + offsetX / dst.width, y, scale)
                }
            }
            return null
        }

        /** Both sides - [pageRect] answers with whichever it finds first, which would half-fade. */
        override fun fadeRect(dst: GPUTexture): FloatArray? {
            var union: FloatArray? = null
            forEachSide { side, offsetX ->
                val image = (side as? ImageSingle)?.currentImage
                if (image != null && image.mipmaps.isNotEmpty()) {
                    val r = image.placement(dst, x + offsetX / dst.width, y, scale)
                    val u = union
                    union = if (u == null) r else floatArrayOf(
                        min(u[0], r[0]), min(u[1], r[1]), max(u[2], r[2]), max(u[3], r[3])
                    )
                }
            }
            return union
        }

        override val backgroundColor: Int?
            get() = left?.backgroundColor ?: right?.backgroundColor

        override fun drawBackgroundColumns(
            pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
        ) = forEachSide { side, srcOffsetX ->
            val color = side.backgroundColor
            if (color != null) {
                val (x1, x2) = sideColumn(side, srcOffsetX, dst)
                Draw.rect(pass, offsetX + x1, offsetY, offsetX + x2, offsetY + 1f, color)
            }
        }

        /**
         * [side]'s left/right edges within [dst], normalised - the whole width when it is the
         * only side. An image side goes through [Image.placement] so its own [Image.x] counts; a
         * [Render] side has no such offset and gets the same formula without it.
         */
        private fun sideColumn(
            side: ImagePage, srcOffsetX: Float, dst: GPUTexture
        ): Pair<Float, Float> {
            if (backgroundSpansFullWidth) return 0f to 1f
            val placeX = x + srcOffsetX / dst.width
            (side as? ImageSingle)?.currentImage?.let { image ->
                val rect = image.placement(dst, placeX, y, scale)
                return rect[0] to rect[2]
            }
            val center = 0.5f + scale * (placeX + WebGpuRenderer.offsetX)
            val half = scale * 0.5f * side.width / dst.width
            return center - half to center + half
        }

        override fun horizontalExtent(): Pair<Float, Float> =
            (left?.width ?: 0).toFloat() to (right?.width ?: 0).toFloat()

        /** Total width (sum of both sides' widths) */
        override val width: Int
            get() = (left?.width ?: 0) + (right?.width ?: 0)

        /** Total height (max of both sides' heights) */
        override val height: Int
            get() = max(left?.height ?: 0, right?.height ?: 0)

        /**
         * Visible width after trim. Inner edges are ignored - a seam is never trimmed - and a
         * [Render] side has no trim at all, so it contributes its full width.
         */
        override val trimWidth: Int
            get() {
                val leftW =
                    leftSingle?.image?.let { it.width - (it.trim?.left ?: 0) } ?: left?.width ?: 0
                val rightW =
                    rightSingle?.image?.let { it.trim?.right ?: it.width } ?: right?.width ?: 0
                return leftW + rightW
            }

        /** Visible height after trim (max of trim heights) */
        override val trimHeight: Int
            get() = max(left?.trimHeight ?: 0, right?.trimHeight ?: 0)

        override val isHalfWidth: Boolean
            get() = true

        /**
         * The sides hang off the seam - the anchor, at [width]/2 - by their own widths, so the
         * span runs [width]/2 - left width to [width]/2 + right width. That equals `0..width`
         * only when both sides are present and equally wide; otherwise it sits off-centre by the
         * difference. Reporting `0..width` regardless let a zoomed-in lone side pan half a page
         * past its own end, and cut off half a page early on the other.
         *
         * Trim plays no part: a seam is never trimmed, and [trimWidth] folds the outer edges in.
         */
        override fun xEdges(trimmed: Boolean): Pair<Float, Float> {
            val (leftWidth, rightWidth) = horizontalExtent()
            val anchor = width / 2f
            return anchor - leftWidth to anchor + rightWidth
        }

        /** Rests with the seam on the viewport centre - see [ImagePage.restingX]. */
        override fun restingX(scale: Float): Float = 0f

        override fun yEdges(trimmed: Boolean): Pair<Int, Int>? {
            if (!trimmed) return null
            val images = listOfNotNull(leftSingle?.image, rightSingle?.image)
            if (images.all { it.trim == null }) return null
            val trimTop = images.mapNotNull { it.trim?.top ?: 0 }.minOrNull() ?: 0
            val trimBottom =
                images.mapNotNull { it.trim?.bottom ?: it.height }.maxOrNull() ?: height
            return trimTop to trimBottom
        }

        /**
         * Each side fit to its own half independently, since the two can differ in size - unlike
         * [ImagePage]'s default, which fits [width]/[height]'s combined span.
         */
        override fun halfWidthScale(halfWidth: Float, parentHeight: Float): Float =
            listOfNotNull(left, right).filter { it.width > 0 && it.height > 0 }
                .minOfOrNull { side ->
                    minOf(halfWidth / side.width, parentHeight / side.height)
                }?.coerceAtLeast(0.01f) ?: 0.01f
    }

    companion object {
        /**
         * Shared scope for fire-and-forget GPU cleanup work.
         * Lives for the application lifetime; individual cleanups are tiny and non-cancellable anyway.
         */
        internal val cleanupScope = CoroutineScope(Dispatchers.Default)

        /** Default [fadeIn] length. */
        const val FADE_MILLIS = 200
    }

    /** True once page content has been decoded/is otherwise ready to draw. */
    open val isDecoded: Boolean get() = false

    /**
     * True once [cleanup] has run and the page's resources are gone or going.
     *
     * Volatile because it is set on whatever thread evicts the page but read on the GPU thread,
     * which uses it to skip drawing a page whose textures are being freed. A render snapshot is
     * captured on the main thread and drawn later, so it can outlive the page it names.
     */
    @Volatile
    var destroyed = false
        private set

    /** True while an animation frame loop owns the current frame. Only ever true for [Images]. */
    open val isAnimated: Boolean get() = false

    /**
     * Incremented each time this page's drawn content changes - an animated [Images] frame, or a
     * [Render] page's [Render.invalidate]. Read by a transition to spot a stale cache slot.
     */
    open val frameVersion: Int get() = 0

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

    /**
     * Draws this page's content instead of blitting an image - see [Render] and [Images]. Given
     * the raw [encoder] rather than an already-open pass, so a page like [Render] that doesn't
     * need [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s stencil masking can open its own
     * pass(es) instead of being forced to match one it has no use for. Must clear [dst] itself -
     * this default does, in its own pass, since [Dummy] (which never overrides this) still needs
     * one: `getCurrentTexture` rotates buffers, so leaving it alone would show stale content from
     * several frames ago. [Render]/[Images] override this to clear as part of their own drawing
     * pass instead of paying for a separate one.
     */
    open fun renderWith(
        encoder: GPUCommandEncoder, x: Float, y: Float, scale: Float, dst: GPUTexture
    ) {
        Draw.clear(encoder, dst, 0)
    }

    /**
     * Draws this page's current live content into [dst] - the paged viewer's per-frame
     * (non-transition) path. Just [renderWith] at home position by default, which already does
     * the right thing for every non-[Images] page (including a no-op [Dummy]); [Images] overrides
     * this to add its [Images.highQuality] tile cache and animated-frame handling.
     *
     * Returns true if the page is now fully covered by sharp tiles, so [ImageViewerState] knows
     * it's safe to prewarm the next page - always false here, since only [Images] has a tile cache.
     */
    internal open fun drawLive(
        encoder: GPUCommandEncoder, dst: GPUTexture, tiles: TileRenderer
    ): Boolean {
        renderWith(encoder, 0f, 0f, 1f, dst)
        return false
    }

    /**
     * As [drawLive], but seeding a [ca.mpreg.webgpuviewer.transition.Transition]'s cache slot
     * instead of the screen - see [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture].
     * Just [renderWith] by default; [Images] overrides this the same way it overrides [drawLive].
     */
    internal open fun renderCacheSeed(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer
    ) {
        renderWith(encoder, 0f, 0f, 1f, tex)
    }

    /**
     * Tile keys newly available to blit since the last call, or null if this page is never tiled
     * (every non-[Images] page, or an [Images] one that isn't [Images.highQuality] or is animated)
     * - see [ca.mpreg.webgpuviewer.renderer.TileRenderer.availableTileKeys].
     */
    internal open fun newlyAvailableTileKeys(tiles: TileRenderer, tex: GPUTexture): Set<Long>? =
        null

    /**
     * Renders this page into a transition's cache slot - see
     * [ca.mpreg.webgpuviewer.transition.Transition.getCachedTexture]. [identityMatches] false
     * means a fresh slot ([renderCacheSeed] from scratch, `LoadOp.Clear`); true means an unchanged
     * one that just needs whatever's newly available layered on (`LoadOp.Load`). Just
     * [renderCacheSeed] by default, since a non-[Images] page never has anything incremental to
     * layer on top of - [Images] overrides this to blit instead when [identityMatches].
     */
    internal open fun renderIntoCache(
        encoder: GPUCommandEncoder, tex: GPUTexture, tiles: TileRenderer, identityMatches: Boolean
    ) {
        renderCacheSeed(encoder, tex, tiles)
    }

    /**
     * This page's own rect within a flat render of it into [dst], as normalised (x1, y1, x2, y2)
     * surface coordinates - for a warp transition ([ca.mpreg.webgpuviewer.transition.TransitionSphere]/
     * flip) to map the page's actual rect rather than treating it as screen-shaped. Null if this
     * page has nothing to draw, which is always true for a non-[Images] page.
     */
    open fun pageRect(dst: GPUTexture): FloatArray? = null

    /**
     * The color a transition should blend/fade toward for this page as a whole - [Images] reads
     * it from its first image's own [ca.mpreg.webgpuviewer.renderer.Image.backgroundColor];
     * [Render] also uses it (when overridden non-null) as its own background fill, both during a
     * transition and for its regular [Render.renderWith]/[Render.renderLoaded] draws - see
     * [Render]'s class doc. Null (nothing to fill/blend toward) by default.
     */
    open val backgroundColor: Int? = null

    /**
     * Draws this page's per-image background colour as separate columns at [offsetX]/[offsetY]
     * within its own cached-surface slide - see
     * [ca.mpreg.webgpuviewer.transition.TransitionBasic] and the
     * [ca.mpreg.webgpuviewer.transition.TransitionStackUp] family. A no-op for a non-[Images] page.
     */
    open fun drawBackgroundColumns(
        pass: GPURenderPassEncoder, dst: GPUTexture, offsetX: Float, offsetY: Float
    ) {
    }

    open val width: Int get() = 0
    open val height: Int get() = 0

    /** As [width]/[height], but after trim - defaults to the untrimmed size. */
    open val trimWidth: Int get() = width
    open val trimHeight: Int get() = height

    /** True if this page uses half-screen layout (dual page or single LEFT/RIGHT). */
    open val isHalfWidth: Boolean get() = false

    /** Left/right edges [minX]/[maxX] pan between: trim's edges once [trimmed], else raw span. */
    protected open fun xEdges(trimmed: Boolean): Pair<Float, Float> = 0f to width.toFloat()

    /**
     * As [xEdges], for top/bottom - null (not the raw fallback) so [nudgedYBounds] can tell real
     * trim edges (margin worth protecting) from the raw image's own (safe to pan past).
     */
    protected open fun yEdges(trimmed: Boolean): Pair<Int, Int>? = null

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

    /**
     * True while a plain (non-zoom) pan fling is actively decaying - set/cleared directly around
     * that decay in [ca.mpreg.webgpuviewer.viewer.ImageViewer]'s gesture handling. Checked there
     * so a tap landing while a fling from the previous gesture is still gliding doesn't also fire
     * [ImageViewerState.onTap] - the tap itself has no motion (that's what makes it a tap, not a
     * drag), so nothing else would otherwise tell the two apart.
     */
    @Volatile
    var isFlinging: Boolean = false

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

    /** How far this page has faded in: 1 is shown, less leaves a [backgroundColor] veil. */
    @Volatile
    var fade: Float = 1f
        private set

    private var fadeJob: Job? = null

    // Written by the thread that decoded the page, read by the draw that attaches it - a stale
    // start of 0 would read as a fade that began at boot.
    @Volatile
    private var fadeMillis = FADE_MILLIS

    @Volatile
    private var fadeStartMillis = 0L

    /** Waiting for a scope: a decode installs the page, the next frame attaches it. */
    @Volatile
    private var fadePending = false

    /**
     * Fade this page in instead of having it appear at once - for one swapped in behind a
     * placeholder that was on screen. Needs a [backgroundColor] to fade from, and runs on the
     * clock from here, so a fade nobody watches is over by the time it is reached.
     */
    @Synchronized
    fun fadeIn(durationMillis: Int = FADE_MILLIS) {
        if (destroyed || backgroundColor == null) return
        fadeMillis = durationMillis
        fadeStartMillis = SystemClock.uptimeMillis()
        fade = 0f
        fadePending = true
        startFade()
    }

    /** Pick the fade up wherever the clock has got to, or skip it if that is already past. */
    @Synchronized
    private fun startFade() {
        if (!fadePending) return
        val scope = scope ?: return
        fadePending = false
        fadeJob?.cancel()

        val elapsed = (SystemClock.uptimeMillis() - fadeStartMillis).toInt()
        if (elapsed >= fadeMillis) {
            fade = 1f
            return
        }
        val from = elapsed.toFloat() / fadeMillis
        fade = from

        fadeJob = scope.launch {
            // No finally: only another fadeIn cancels this, and it sets fade itself.
            animate(from, 1f, animationSpec = tween(fadeMillis - elapsed)) { value, _ ->
                fade = value
                // invalidate(), not onInvalidate: its frameVersion bump re-seeds a transition's
                // cached copy, which would otherwise hold the veil at whatever it was seeded with.
                invalidate()
            }
        }
    }

    /** What [drawFade] veils - [pageRect], except a spread takes both sides (its own override). */
    protected open fun fadeRect(dst: GPUTexture): FloatArray? = pageRect(dst)

    /** Veil this page's rect, in fractions of the target, with what is left of the fade. */
    internal fun drawFade(
        pass: GPURenderPassEncoder,
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        masked: Boolean = true
    ) {
        if (fade >= 1f) return
        val color = backgroundColor ?: return
        val alpha = (((color ushr 24) and 0xFF) * (1f - fade)).toInt().coerceIn(0, 255)
        val veil = (alpha shl 24) or (color and 0xFFFFFF)
        // Only a live draw's pass has the stencil attachment drawMaskedRect's pipeline declares.
        if (masked) RenderPage.drawMaskedRect(pass, x1, y1, x2, y2, veil)
        else Draw.rect(pass, x1, y1, x2, y2, veil)
    }

    /** True while the viewer is drawing this page, itself or as a side of a spread. */
    val isOnScreen: Boolean
        get() = parent?.isOnScreen(this) == true

    /** True when drawing this page draws [other] - itself, or a side [ImageSpread] overrides in. */
    internal open fun covers(other: ImagePage): Boolean = this === other

    /**
     * Adopts this page into [parent]'s viewer. [ImageSpread] passes it on to its sides, which the
     * viewer never fetches itself but which still need a scope to animate in and a way back to
     * the screen.
     */
    internal open fun attach(
        parent: ImageViewerState, scope: CoroutineScope?, onInvalidate: () -> Unit
    ) {
        if (this.parent !== parent) this.parent = parent
        if (this.scope !== scope) this.scope = scope
        if (this.onInvalidate !== onInvalidate) this.onInvalidate = onInvalidate
        if (fadePending) startFade()
    }

    /**
     * Redraws this page, if it is on screen to redraw. [Render] and [Images] also bump
     * [frameVersion] either way, so a page turn re-seeds the cached copy it is sampling rather
     * than showing the content as it was when the turn began.
     */
    open fun invalidate() {
        if (isOnScreen) onInvalidate?.invoke()
    }

    private val parentWidth: Float
        get() = parent?.width?.toFloat() ?: 0f

    private val parentHeight: Float
        get() = parent?.viewportHeight ?: 0f

    /**
     * [isHalfWidth]'s fit scale: each side of a spread can be a differently sized image, so
     * [Images] overrides this to fit each one independently rather than [width]/[height]'s
     * combined span - the default here is only ever exercised by a non-spread page.
     */
    protected open fun halfWidthScale(halfWidth: Float, parentHeight: Float): Float =
        minOf(halfWidth / width, parentHeight / height).coerceAtLeast(0.01f)

    val atHome: Boolean
        get() = x.closeTo(homeX) && y.closeTo(homeY) && atHomeScale

    val atHomeScale: Boolean
        get() = scale.closeTo(homeScale)

    var homeScale: Float = -1f
        get() {
            if (field > 0) return field

            if (parentWidth <= 0f || parentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen, no trim
                return halfWidthScale(parentWidth / 2f, parentHeight)
            }

            // Single SINGLE page: fit trim to full screen
            val w = trimWidth.toFloat().takeIf { it > 0f } ?: return 0.01f
            val h = trimHeight.toFloat().takeIf { it > 0f } ?: return 0.01f
            return minOf(parentWidth / w, parentHeight / h).coerceAtLeast(0.01f)
        }

    var homeX: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxX(scale).fastCoerceIn(minX(scale), maxX(scale))
        }

    var homeY: Float = 0f
        get() {
            if (field != 0f) return field
            val scale = homeScale
            return maxY(scale).fastCoerceIn(minY(scale), maxY(scale))
        }

    var minScale = 0f
        get() {
            if (field > 0) return field
            if (parentWidth <= 0f || parentHeight <= 0f) return 0.01f

            if (isHalfWidth) {
                // Half-width layout: each image fits in half screen
                return halfWidthScale(parentWidth / 2f, parentHeight)
            }

            // Single SINGLE page
            return minOf(parentWidth / width, parentHeight / height).coerceAtLeast(0.01f)
        }

    var maxScale = 0f
        get() = if (field > 0) field else max(doubleTapScale * 2, 2f)

    var doubleTapScale: Float = 0f
        get() = if (field != 0f) field else max(minScale, homeScale) * 2

    // BOUNDS:
    // cutout ignore:
    //  with trim:
    //      >= homeScale: viewport pan over trimmed content
    //      < homeScale: viewport pan over untrimmed content
    //  without trim: viewport pan over content
    //
    // cutout avoid:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: nudge below cutout
    //
    // cutout shift:
    //  with trim:
    //      >= homeScale: cut viewport pan over trimmed content
    //      < homeScale: cut viewport pan over untrimmed content
    //  without trim: cut viewport pan over content
    //  if content is fully visible: center in cut viewport

    /**
     * As [xBounds] but never collapsed (min > max when there's slack). [nudgedYBounds] needs
     * the true floor - the collapsed center sits below it, letting panning reveal past it.
     */
    private fun rawBounds(
        size: Int, nearEdge: Float, farEdge: Float, parentSize: Int, scale: Float
    ): Pair<Float, Float> {
        val maxV = (0.5f * size - nearEdge) / parentSize - 0.5f / scale
        val minV = (0.5f * size - farEdge) / parentSize + 0.5f / scale
        return minV to maxV
    }

    /** [minX]/[maxX] together, computing [homeScale] and [rawBounds] only once per call. */
    private fun xBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val (left, right) = xEdges(scale >= homeScale)
        val (minV, maxV) = rawBounds(width, left, right, parent.width, scale)
        if (minV > maxV) {
            val rest = restingX(scale)
            return rest to rest
        }
        return minV to maxV
    }

    /**
     * Where the page rests horizontally once its content fits the viewport - the middle of
     * [xEdges]'s span, so the content sits centred.
     *
     * [ImageSpread] overrides it to rest on the seam instead. That is what keeps a lone left/right
     * side in its own half; centring the span - what the midpoint gives once one side is missing -
     * would pull it to the middle, indistinguishable from a page with no partner.
     */
    protected open fun restingX(scale: Float): Float {
        val parent = parent ?: return 0f
        val (near, far) = xEdges(scale >= homeScale)
        return (0.5f * width - 0.5f * (near + far)) / parent.width
    }

    fun minX(scale: Float): Float = xBounds(scale).first

    fun maxX(scale: Float): Float = xBounds(scale).second

    /**
     * "Ignore": [xBounds]'s plain collapse-when-it-fits.
     *
     * "Avoid"/"shift": near/top bound pushed further by the *full* [ImageViewerState.cutoutTopPx]
     * so the cut viewport (real viewport minus the cutout) can pan over all the content - unless
     * still fully visible there even after the push, which collapses to one rest point instead of
     * a pointless range: "shift" always rests centered in the cut viewport (half the push, since
     * shrinking the viewport only moves its center by half); "avoid" only nudges - just far enough
     * to clear the cutout, not all the way to centered-in-cut-viewport - and only when the plain
     * whole-screen-centered rest would actually overlap the cutout.
     */
    private fun nudgedYBounds(scale: Float): Pair<Float, Float> {
        val parent = parent ?: return 0f to 0f
        val trimmed = scale >= homeScale
        val (top, bottom) = yEdges(trimmed) ?: (0 to height)
        val (floor, natMax) = rawBounds(
            height, top.toFloat(), bottom.toFloat(), parent.height, scale
        )
        val slack = floor > natMax
        val center = (floor + natMax) / 2f

        if (!parent.avoidCutout || parent.cutoutTopPx <= 0f || parent.height <= 0) {
            return if (slack) center to center else floor to natMax
        }

        val fullPush =
            if (isHalfWidth && !trimmed) 0f else parent.cutoutTopPx / (scale * parent.height)
        val pushed = natMax + fullPush

        if (slack) {
            // "pushed" (natMax + fullPush) is exactly the rest position where the near edge sits
            // flush against the cutout - so center < pushed means the plain centered rest would
            // sit above that line, i.e. under the cutout.
            val overlapsCutout = center < pushed
            if (!parent.alwaysAvoidCutout && !overlapsCutout) return center to center
            val rest = if (parent.alwaysAvoidCutout) center + fullPush / 2f else pushed
            if (rest < floor) return rest to rest
        }
        return floor to pushed
    }

    fun minY(scale: Float): Float = nudgedYBounds(scale).first

    fun maxY(scale: Float): Float = nudgedYBounds(scale).second

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

        val minX = minX(targetScale)
        val maxX = maxX(targetScale)
        val minY = minY(targetScale)
        val maxY = maxY(targetScale)

        val scaleChanging = targetScale != startScale
        val diffEnd = if (scaleChanging) 1 / targetScale - 1 / startScale else 1f

        val endX = when {
            origin != null && scaleChanging -> (startX + (origin.x - 0.5f) * diffEnd).fastCoerceIn(
                minX, maxX
            )

            origin != null -> x.fastCoerceIn(minX, maxX)
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
    open fun cleanup() {
        if (destroyed) return
        destroyed = true

        animationJob?.cancel()
        animationJob = null
        fadeJob?.cancel()
        fadeJob = null
        fadePending = false
    }

    /** Alias for cleanup() */
    fun destroy() = cleanup()
}
