package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.util.fastCoerceIn
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.solveImagePlacement
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlin.math.max

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    companion object {
        const val MAX_VISIBLE_PAGES = 4
    }

    var scale = 1f

    var offsetX = 0f

    /**
     * How much of the viewport width a page fills when fully zoomed out, from 0 to 1. The
     * default 1 zooms out to exactly the full width; 0.6 stops with the page at 60% of it and
     * margin either side.
     *
     * Only the zoom-out floor moves. A page is still laid out and measured against the full
     * width - [getPageHeight] and the whole document coordinate space are unchanged - so this
     * decides how far out a pinch may go, not how tall anything is.
     *
     * Clamped away from 0, which is not a scale anything can be drawn at. Setting it lifts a
     * [scale] that is now below the floor, so it takes effect without waiting for a gesture.
     */
    var minZoomWidthFraction: Float = 1f
        set(value) {
            val clamped = value.fastCoerceIn(0.01f, 1f)
            if (clamped == field) return
            field = clamped
            if (scale < clamped) scale = clamped
            invalidate()
        }

    /** Lowest [scale] a gesture may settle at - see [minZoomWidthFraction]. */
    val minScale: Float get() = minZoomWidthFraction

    /** Follows [minScale], so a double tap off the zoom-out floor still doubles what is on screen. */
    val doubleTapScale: Float get() = minScale * 2f

    val maxScale: Float get() = max(doubleTapScale * 2f, 4f)

    /**
     * True while [ImageViewerContinuous]'s gestures are actively driving zoom (pinch, drag, fling,
     * snap-back). Gates every visible page's tile grid here the same way [ImagePage.isScaleAnimating]
     * gates the paged viewer's.
     */
    @Volatile
    var isScaleAnimating: Boolean = false

    /**
     * True while a plain (non-zoom) fling is actively scrolling. Generating a filtered tile is
     * real GPU work sharing the render thread with the frame itself, so doing it while the camera
     * is moving fast under its own momentum both wastes the work (the content is about to scroll
     * back out of view) and is a real source of visible frame lag. Combined with
     * [isScaleAnimating] wherever a caller needs "don't generate right now" - kept separate here
     * since the two are driven by different gestures and one may be true without the other.
     */
    @Volatile
    var isFlinging: Boolean = false

    private val scrollLock = Any()

    var scrollY = 0f
        private set

    /**
     * Visual-only slide, animated to 0 by [animateSlideIn]. Kept out of [scrollY], which would
     * walk into the page before it and report a page change of its own.
     */
    private var slideOffset = 0f

    /**
     * Layout height of [page] in screen pixels.
     *
     * Measured the same way decoded or not: a placeholder carrying the real aspect ratio has to
     * occupy exactly the space its decoded self will, or the pages below jump when it decodes.
     * The guard is only for pages with no width to fit against, which have no ratio to scale by.
     *
     * Only an [ImagePage.ImageSingle] (which [ImagePage.ImageSpread] also is) fits the viewer's
     * full width - this mode's reading convention for raster content. A [ImagePage.Render] page's
     * width/height are the author's deliberate choice, not something to stretch, so it is
     * reserved and drawn at its native size - see the matching pageScale in [renderSnapshot].
     */
    fun getPageHeight(page: ImagePage): Float {
        if (page !is ImagePage.ImageSingle) return page.height.toFloat()
        val pageWidth = page.width
        if (pageWidth <= 0) return page.height.toFloat()
        return page.height * (width.toFloat() / pageWidth)
    }

    /** Height page 0 was last measured at, to carry the position across a decode correcting it. */
    private var currentPageHeight: Float? = null

    /**
     * The page read through, reported when it changes: the deepest one whose bottom has reached
     * the viewport's, or that covers its top. Where [onPageChange] means "reached this page's
     * top", this means "read past it". Observation only - nothing here moves the scroll.
     */
    var onPageScrolledThrough: ((ImagePage) -> Unit)? = null

    private var lastScrolledThrough: ImagePage? = null

    /**
     * Pages the last frame reached below and above the current one. What the viewport actually
     * shows depends on the zoom, so a caller's decode window has to follow this rather than a
     * fixed count - a page on screen has to be decoded, not merely reserved.
     */
    @Volatile
    var pagesBelow: Int = 0
        private set

    @Volatile
    var pagesAbove: Int = 0
        private set

    /**
     * Document-space top of whatever page currently sits at [scrollY] == 0, in screen pixels at
     * zoom 1. The only state the continuous coordinate space needs to persist across
     * frames: every other visible page's position is re-derived fresh each frame from this one
     * value (see [captureRenderState]'s walk), rather than stored per page - a page's identity
     * isn't stable across a decode (the app hands over a new object), so anything kept on the
     * page itself would be silently lost exactly when a placeholder corrects to its real height.
     * Updated only here, in [scrollBy], using the height of whichever page is actually being
     * crossed.
     */
    private var anchorDocY = 0f

    /**
     * Scroll by [deltaPixels], moving the current page as many times as the delta covers.
     *
     * A single fling frame can cross more than one page when pages are short, so both walks
     * loop. Each also stops on a zero-height page, which would otherwise never advance the
     * position and spin here forever.
     */
    fun scrollBy(deltaPixels: Float) {
        synchronized(scrollLock) {
            getPage(0) ?: return
            slideOffset = 0f

            scrollY += deltaPixels

            // Backwards, while the position sits above the top of the current page.
            while (scrollY < 0) {
                if (getPage(-1) == null) {
                    scrollY = 0f
                    break
                }
                onPageChange?.invoke(-1)
                val newPage = getPage(0) ?: return
                val newHeight = getPageHeight(newPage)
                anchorDocY -= newHeight
                currentPageHeight = newHeight
                // No height to hold a position inside, so rest at its top rather than leave the
                // position above it, which the next scroll would read as another step back.
                if (newHeight <= 0f) {
                    scrollY = 0f
                    break
                }
                scrollY += newHeight
            }

            // Forwards, while it sits past the bottom of it. Stops at the last page rather than
            // stepping off the end, which would leave the position short instead of clamping.
            while (true) {
                val page = getPage(0) ?: return
                val pageHeight = getPageHeight(page)
                if (scrollY <= pageHeight || pageHeight <= 0f) break
                if (getPage(1) == null) {
                    scrollY = pageHeight
                    break
                }
                onPageChange?.invoke(1)
                anchorDocY += pageHeight
                val newPage = getPage(0) ?: return
                currentPageHeight = getPageHeight(newPage)
                scrollY -= pageHeight
            }

            clampToDocumentEnd()
        }
    }

    /**
     * Furthest [scrollY] may go: the last page's bottom stops at the viewport's, never above it.
     * Null when the document doesn't end within the pages this mode draws, so nothing to clamp.
     * Negative when the end falls above page 0's own top - see [clampToDocumentEnd].
     */
    private fun maxScrollY(): Float? {
        val viewportHeight = height / scale
        var bottom = 0f
        for (i in 0..MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: return bottom - viewportHeight
            val pageHeight = getPageHeight(page)
            if (pageHeight <= 0f) break
            bottom += pageHeight
            // Enough content below to fill the viewport, whatever follows it.
            if (bottom - viewportHeight > scrollY) break
        }
        return null
    }

    /**
     * Hold [scrollY] at the end of the document, which the walks above can overshoot. A last page
     * shorter than the viewport ends above page 0's own top, and [scrollY] can't hold a negative -
     * the backward walk reads that as "step to the page above" - so step back to a page that can.
     */
    private fun clampToDocumentEnd() {
        while (true) {
            val max = maxScrollY() ?: return
            if (scrollY <= max) return
            if (max >= 0f) {
                scrollY = max
                return
            }
            // Nothing above to measure from, so the document's top is as far as this goes.
            if (getPage(-1) == null) {
                scrollY = 0f
                return
            }
            onPageChange?.invoke(-1)
            val newPage = getPage(0) ?: return
            val newHeight = getPageHeight(newPage)
            anchorDocY -= newHeight
            currentPageHeight = newHeight
            // No height yet to hold it either, so rest at its top.
            if (newHeight <= 0f) {
                scrollY = 0f
                return
            }
            // The same document position, measured off the page now at 0.
            scrollY = max + newHeight
        }
    }

    /**
     * Document-space position of the viewport's top, in page-space pixels at zoom 1. Where the
     * reader is in a form that survives a page crossing, which [scrollY] on its own doesn't - so
     * it is what to remember a position by, and [scrollTo] what to put it back with.
     */
    val documentY: Float get() = synchronized(scrollLock) { anchorDocY + scrollY }

    /** Put the viewport's top at [docY] - see [documentY]. */
    fun scrollTo(docY: Float) {
        synchronized(scrollLock) { scrollBy(docY - (anchorDocY + scrollY)) }
    }

    /** Move to the top of the page [getPage] now answers 0 with, after the app jumps pages. */
    fun resetScroll() {
        synchronized(scrollLock) {
            scrollY = 0f
            // A different page now: its own height is the baseline, not the page left behind.
            currentPageHeight = null
        }
    }

    /** Slide the current page into place after a jump - [direction] 1 when it came from below. */
    fun animateSlideIn(direction: Int) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            try {
                animate(
                    direction * height / 2f, 0f, animationSpec = spring(
                        stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.5f
                    )
                ) { value, _ ->
                    slideOffset = value
                    invalidate()
                }
            } finally {
                // Not if cancelled: this resumes after whatever replaced it set its own.
                if (animationJob === coroutineContext[Job]) {
                    slideOffset = 0f
                    invalidate()
                }
            }
        }
    }

    fun animateScroll(deltaPixels: Float) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            var lastValue = 0f
            animate(
                0f, deltaPixels, animationSpec = spring(
                    stiffness = Spring.StiffnessMediumLow, visibilityThreshold = 0.002f
                )
            ) { value, _ ->
                scrollBy(value - lastValue)
                lastValue = value
                invalidate()
            }
        }
    }

    /** One page visible this frame, with the document-space top [captureRenderState] found it at. */
    private class VisiblePage(val page: ImagePage, val docTop: Float, val pageHeight: Float)

    private class ContinuousRenderSnapshot(
        val pages: List<VisiblePage>,
        val scale: Float,
        val offsetX: Float,
        /** Document position (see [anchorDocY]) currently at the viewport's vertical centre. */
        val cameraDocY: Float,
        /** [isScaleAnimating] or [isFlinging] - either means "don't generate tiles right now". */
        val suppressGeneration: Boolean,
    )

    override fun captureRenderState(): Any = synchronized(scrollLock) {
        val screenH = height.toFloat()

        val page0 = getPage(0)
        if (page0 != null) {
            val pageHeight = getPageHeight(page0)
            // A decode correcting a placeholder's height holds the same fraction of the page: at
            // its top nothing moves, near its bottom the pages below stay put. Both heights have
            // to be measured, and an unmeasured one is not a baseline to correct against later.
            currentPageHeight?.let { h -> if (h > 0f && pageHeight > 0f) scrollY *= pageHeight / h }
            if (pageHeight > 0f) currentPageHeight = pageHeight
            // A decode shortening the document under a position already at its end: only
            // [scrollBy] used to notice, on the next scroll, as a jump.
            clampToDocumentEnd()
        }

        // After the clamp, which can step the page at 0 back.
        val y0 = if (page0 != null) -scrollY + slideOffset else 0f

        // Document position at the viewport's centre - the point both the fast path and
        // TileRenderer's continuous overloads zoom around, so they always agree on where a page
        // belongs.
        val cameraDocY = anchorDocY - y0 + 0.5f * screenH

        val pages = mutableListOf<VisiblePage>()

        // Visible band in unscaled page space. Zoom is centered on the screen, so the
        // viewport covers screenH / scale of page space around the screen center.
        val visTop = 0.5f * screenH - screenH / (2f * scale)
        val screenBot = 0.5f * screenH + screenH / (2f * scale)
        // +1 tile of margin, matching TileRenderer's own prefetch ring, so a boundary tile just
        // past the viewport has its page already discovered.
        val visBot = screenBot + tiles.preferredTileSize / scale

        // Read past, not merely reached - see [onPageScrolledThrough]. No height, no reading.
        fun isScrolledThrough(top: Float, pageHeight: Float) =
            pageHeight > 0f && (top + pageHeight <= screenBot || top < visTop)

        var scrolledThrough: ImagePage? = null

        // Backward: pages above page 0, needed once zoomed out enough that visTop goes negative -
        // i.e. the visible band reaches above where page 0 itself starts. Mirrors the forward
        // walk below, just toward negative indices.
        var yTop = y0
        var iBack = -1
        var docTopBack = anchorDocY
        var above = 0
        while (yTop > visTop && iBack >= -MAX_VISIBLE_PAGES) {
            val page = getPage(iBack) ?: break
            above = -iBack
            val pageHeight = getPageHeight(page)
            docTopBack -= pageHeight
            yTop -= pageHeight
            // Walking up, so the first match is the deepest one above page 0.
            if (scrolledThrough == null && isScrolledThrough(yTop, pageHeight)) scrolledThrough =
                page
            // Walked upward, so each one goes in front of the last - top to bottom, as the
            // forward walk below appends.
            if (page.isDecoded) {
                pages.add(0, VisiblePage(page, docTopBack, pageHeight))
            }
            if (pageHeight <= 0f) break
            iBack--
        }

        // Walk forward until the viewport (plus margin) is covered or MAX_VISIBLE_PAGES
        // is reached, whichever comes first - zoomed out far enough (or with short enough pages),
        // the document-space bound alone would keep walking past it.
        // Purely local: nothing is written back to a page, so only [anchorDocY] needs to survive
        // across frames for this to stay correct.
        var y = y0
        var i = 0
        var docTop = anchorDocY
        var prevHeight = 0f
        var hasPrev = false
        var below = 0
        while (y < visBot && i <= MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: break
            below = i
            // Anchor to the previous page in this walk, never frozen: an undecoded page's height
            // is a guess, so re-deriving this fresh every frame self-corrects once it decodes.
            if (hasPrev) docTop += prevHeight
            hasPrev = true
            val pageHeight = getPageHeight(page)

            // Walking down, so a later match replaces whatever the backward walk found.
            if (isScrolledThrough(y, pageHeight)) scrolledThrough = page

            if (y + pageHeight > visTop && page.isDecoded) {
                pages.add(VisiblePage(page, docTop, pageHeight))
            }

            // A zero-height page never advances y, so stop rather than ask for pages forever.
            if (pageHeight <= 0f) break

            prevHeight = pageHeight
            y += pageHeight
            i++
        }

        onScreenPages = pages.map { it.page }
        pagesBelow = below
        pagesAbove = above

        // By identity: a page that stays the deepest one read through is reported once.
        scrolledThrough?.takeIf { it !== lastScrolledThrough }?.let {
            lastScrolledThrough = it
            onPageScrolledThrough?.invoke(it)
        }

        ContinuousRenderSnapshot(pages, scale, offsetX, cameraDocY, isScaleAnimating || isFlinging)
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder, texture: GPUTexture, snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        tiles.newFrame()
        if (s.pages.isEmpty()) return

        // ImagePage.ImageSingle pages batch into one shared pass (they never overlap vertically, so one
        // clear plus one draw per image writes each pixel once). A Render page (ImagePage.Render,
        // e.g. a loading placeholder) can't join that batch - it has no image/tile to draw, only
        // its own render() - so it draws afterward via its own renderLoaded call instead.
        // renderLoaded (unlike renderWith) loads rather than clears its pass, since this texture
        // is shared with every other visible page and clearing it would blank them too - which
        // relies on something having cleared the texture first. The ImageSingle batch's renderPass
        // does that whenever there is one; if every visible page turns out to be a Render page,
        // [ca.mpreg.webgpuviewer.draw.clear] does it instead so a Render page never has to paint
        // over stale content from prior frames.
        val hasImagePage = s.pages.any { it.page is ImagePage.ImageSingle }

        val dstW = texture.width.toFloat()
        val dstH = texture.height.toFloat()
        // Screen position of document space's origin - mirrors TileRenderer's continuous anchor
        // exactly, so the fast path, tile cache, and Render pages below all agree on placement.
        val anchorX = dstW / 2f + s.scale * (s.offsetX * dstW + WebGpuRenderer.offsetX * dstW)
        val anchorY = dstH / 2f - s.scale * s.cameraDocY + s.scale * WebGpuRenderer.offsetY * dstH

        if (hasImagePage) {
            renderPass(encoder, texture) { pass ->
                s.pages.forEach { vp ->
                    val page = vp.page as? ImagePage.ImageSingle ?: return@forEach
                    // The snapshot was captured on the main thread; the page can have been
                    // evicted since, in which case its images' buffers are gone and drawing one
                    // throws.
                    if (page.destroyed || !page.isDecoded || page.width <= 0) return@forEach

                    val pageScale = dstW / page.width

                    // Tiles first, marking the stencil; the sampler below shades only what is
                    // left, and nothing at all once the draw reports full coverage. Animated
                    // pages never get tiles, so they skip the call outright.
                    val covered = !page.isAnimated && tiles.draw(
                        pass,
                        page,
                        texture,
                        s.cameraDocY,
                        vp.docTop,
                        s.offsetX,
                        s.scale,
                        s.suppressGeneration
                    )
                    if (!covered) {
                        val imageScale = pageScale * s.scale
                        page.forEachImage { image, srcOffsetX ->
                            if (image.mipmaps.isEmpty()) return@forEachImage
                            val docCenterX = pageScale * (srcOffsetX + image.x)
                            val docCenterY = vp.docTop + 0.5f * vp.pageHeight + pageScale * image.y
                            val targetX = anchorX + s.scale * docCenterX
                            val targetY = anchorY + s.scale * docCenterY
                            val (x, y) = solveImagePlacement(
                                targetX, targetY, imageScale, image, dstW, dstH
                            )
                            // Content not worth linear-light correctness
                            // (ImagePage.ImageSingle.highQuality) gets the plain sampler - it never
                            // reaches the tile cache either. Animated pages are never highQuality
                            // but always want the fast sampler regardless, since they swap images
                            // every frame. Both are stencil-tested against the tile draw above,
                            // skipping pixels it already covered.
                            if (page.isAnimated || page.highQuality) {
                                RenderPage.renderFast(pass, image, texture, x, y, imageScale)
                            } else {
                                RenderPage.renderFast(
                                    pass, image, texture, x, y, imageScale, linear = false
                                )
                            }
                        }
                    }

                    // After everything that draws the page, and over its own band only: pages
                    // tile vertically, so veiling anything wider would fade its neighbours too.
                    if (page.fade < 1f) {
                        val top = anchorY + s.scale * vp.docTop
                        page.drawFade(
                            pass,
                            (anchorX - s.scale * dstW / 2f) / dstW,
                            top / dstH,
                            (anchorX + s.scale * dstW / 2f) / dstW,
                            (top + s.scale * vp.pageHeight) / dstH
                        )
                    }
                }
            }
        } else {
            Draw.clear(encoder, texture, 0)
        }

        s.pages.forEach { vp ->
            if (vp.page is ImagePage.ImageSingle) return@forEach
            // Only ImagePage.ImageSingle overrides isDecoded to anything other than this fixed
            // "Render (or a subclass) with drawable content" default - the isDecoded filter in
            // captureRenderState already excludes anything else (e.g. Dummy).
            val page = vp.page as ImagePage.Render
            if (page.destroyed) return@forEach

            // Render's own render(dst, x, y, scale) convention has no fit-to-width factor to
            // undo - x/y/scale there are already fractions of dst (screen) size, not of this
            // page's own declared width/height (see getPageHeight's doc: unlike an Images page,
            // this page's size is never stretched to the viewer's width). So the only screen
            // scale in play is the pinch-zoom (s.scale) times whatever this page's own scale is -
            // folding in a dstW/page.width factor here (as an Images page's placement does) would
            // scale its content by that ratio for no reason, which is exactly what "overdrawing
            // its size" looked like. page.x/page.y are left out of the position for the same
            // reason: they're in that dst-fraction unit, not vp.docTop's doc-space-pixel one, so
            // they can't be combined with it - harmless since a locked-scale page like
            // TransitionPage never has them set to anything but 0 anyway.
            val renderScale = s.scale * page.scale
            val targetX = anchorX
            val targetY = anchorY + s.scale * (vp.docTop + 0.5f * vp.pageHeight)

            val x = (targetX - dstW / 2f) / (renderScale * dstW)
            val y = (targetY - dstH / 2f) / (renderScale * dstH)
            page.renderLoaded(encoder, x, y, renderScale, texture)
        }
    }
}
