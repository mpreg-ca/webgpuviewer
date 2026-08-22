package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.renderer.TileRenderer.Companion.TILE_SIZE
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.solveImagePlacement
import kotlinx.coroutines.launch

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    companion object {
        private const val MAX_VISIBLE_PAGES = 2
    }

    var scale = 1f

    var offsetX = 0f

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

    /**
     * Layout height of [page] in screen pixels.
     *
     * Measured the same way whether or not the page has decoded yet: a placeholder that carries
     * the real aspect ratio has to occupy exactly the space its decoded self will, or the pages
     * below it jump the moment it decodes. The guard is only for pages with no width to fit
     * against (all images null), which have no aspect ratio to scale by.
     */
    fun getPageHeight(page: ImagePage): Float {
        val pageWidth = page.width
        if (pageWidth <= 0) return page.height.toFloat()
        return page.height * (width.toFloat() / pageWidth)
    }

    var currentPageHeight: Float? = null

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
                if (newHeight <= 0f) break
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

            if (getPage(1) == null) {
                val page = getPage(0) ?: return
                scrollY = scrollY.coerceAtMost(getPageHeight(page))
            }
        }
    }

    fun animateScroll(deltaPixels: Float) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            var lastValue = 0f
            animate(
                0f, deltaPixels, animationSpec = spring(
                    stiffness = Spring.StiffnessMediumLow,
                    visibilityThreshold = 0.002f
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

        val y0 = getPage(0)?.let { page ->
            val pageHeight = getPageHeight(page)
            if (currentPageHeight != pageHeight && scrollY > 0) {
                currentPageHeight?.let { h -> scrollY -= pageHeight - h }
                currentPageHeight = pageHeight
            }
            -scrollY
        } ?: 0f

        // Document position at the viewport's centre - the point both the fast path and
        // TileRenderer's continuous overloads zoom around, so they always agree on where a page
        // belongs.
        val cameraDocY = anchorDocY - y0 + 0.5f * screenH

        val pages = mutableListOf<VisiblePage>()

        // Visible band in unscaled page space. Zoom is centered on the screen, so the
        // viewport covers screenH / scale of page space around the screen center.
        val visTop = 0.5f * screenH - screenH / (2f * scale)
        // +1 tile of margin, matching TileRenderer's own prefetch ring, so a boundary tile just
        // past the viewport has its page already discovered.
        val visBot = 0.5f * screenH + screenH / (2f * scale) + TILE_SIZE / scale

        // Backward: pages above page 0, needed once zoomed out enough that visTop goes negative -
        // i.e. the visible band reaches above where page 0 itself starts. Mirrors the forward
        // walk below, just toward negative indices.
        var yTop = y0
        var iBack = -1
        var docTopBack = anchorDocY
        val backPages = mutableListOf<VisiblePage>()
        while (yTop > visTop && iBack >= -MAX_VISIBLE_PAGES) {
            val page = getPage(iBack) ?: break
            val pageHeight = getPageHeight(page)
            docTopBack -= pageHeight
            yTop -= pageHeight
            if (page.images.any { it != null }) {
                backPages.add(VisiblePage(page, docTopBack, pageHeight))
            }
            if (pageHeight <= 0f) break
            iBack--
        }
        pages.addAll(backPages.asReversed())

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
        while (y < visBot && i <= MAX_VISIBLE_PAGES) {
            val page = getPage(i) ?: break
            // Anchor to the previous page in this walk, never frozen: an undecoded page's height
            // is a guess, so re-deriving this fresh every frame self-corrects once it decodes.
            if (hasPrev) docTop += prevHeight
            hasPrev = true
            val pageHeight = getPageHeight(page)

            if (y + pageHeight > visTop && page.images.any { it != null }) {
                pages.add(VisiblePage(page, docTop, pageHeight))
            }

            // A zero-height page never advances y, so stop rather than ask for pages forever.
            if (pageHeight <= 0f) break

            prevHeight = pageHeight
            y += pageHeight
            i++
        }

        ContinuousRenderSnapshot(pages, scale, offsetX, cameraDocY, isScaleAnimating || isFlinging)
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        tiles.newFrame()
        if (s.pages.isEmpty()) return

        // Every visible page shares one render pass. Pages never overlap vertically, so the clear
        // plus one draw per image writes each pixel once, with no per-page attachment
        // load/store.
        renderPass(encoder, texture) { pass ->
            val dstW = texture.width.toFloat()
            val dstH = texture.height.toFloat()
            // Screen position of document space's origin - mirrors TileRenderer's continuous
            // anchor exactly, so the fast path and tile cache always agree on image placement.
            val anchorX = dstW / 2f + s.scale * (s.offsetX * dstW + WebGpuRenderer.offsetX * dstW)
            val anchorY =
                dstH / 2f - s.scale * s.cameraDocY + s.scale * WebGpuRenderer.offsetY * dstH

            s.pages.forEach { vp ->
                val page = vp.page
                // The snapshot was captured on the main thread; the page can have been evicted
                // since, in which case its images' buffers are gone and drawing one throws.
                if (page.destroyed || page.images.all { it == null } || page.width <= 0) return@forEach

                val pageScale = dstW / page.width

                // Tiles draw first, marking every pixel they cover in the stencil buffer
                // tiles.draw() writes to; the sampler shader below then only shades what's left
                // uncovered instead of the whole viewport - skipped entirely once tiles alone
                // already cover this page.
                tiles.draw(
                    pass,
                    page,
                    texture,
                    s.cameraDocY,
                    vp.docTop,
                    s.offsetX,
                    s.scale,
                    s.suppressGeneration
                )

                val covered = tiles.isFullyCovered(
                    page, texture, s.cameraDocY, vp.docTop, s.offsetX, s.scale, s.suppressGeneration
                )
                if (!covered) {
                    val imageScale = pageScale * s.scale
                    page.images.forEach { image ->
                        image ?: return@forEach
                        if (image.mipmaps.isEmpty()) return@forEach
                        // Same convention as RenderPage/TileRenderer's spread offset, keyed off
                        // the image's own position rather than its list index, so a spread
                        // renders identically here and in the paged viewer.
                        val spreadShiftDocX = when (image.position) {
                            Image.Position.LEFT -> -0.5f * image.width * pageScale
                            Image.Position.RIGHT -> 0.5f * image.width * pageScale
                            Image.Position.SINGLE -> 0f
                        }
                        val docCenterX = spreadShiftDocX + pageScale * image.x
                        val docCenterY = vp.docTop + 0.5f * vp.pageHeight + pageScale * image.y
                        val targetX = anchorX + s.scale * docCenterX
                        val targetY = anchorY + s.scale * docCenterY
                        val (x, y) = solveImagePlacement(
                            targetX,
                            targetY,
                            imageScale,
                            image,
                            dstW,
                            dstH
                        )
                        // Content not worth linear-light correctness (ImagePage.highQuality)
                        // gets the plain sampler instead - it never reaches the tile cache either.
                        // Both are stencil-tested against the tile draw above, skipping pixels it
                        // already covered.
                        if (page.highQuality) {
                            RenderPage.renderFast(pass, image, texture, x, y, imageScale)
                        } else {
                            RenderPage.renderPlainMasked(pass, image, texture, x, y, imageScale)
                        }
                    }
                }
            }
        }
    }
}
