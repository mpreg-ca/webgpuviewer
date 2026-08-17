package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.RenderPage
import kotlinx.coroutines.launch

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    var scale = 1f

    var offsetX = 0f

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
                currentPageHeight = newHeight
                if (newHeight <= 0f) break
                scrollY += newHeight
            }

            // Forwards, while it sits past the bottom of it. Stopping at the last page rather
            // than stepping off the end: with no page to move to, the step would leave the
            // position short by a page height instead of clamping.
            while (true) {
                val page = getPage(0) ?: return
                val pageHeight = getPageHeight(page)
                if (scrollY <= pageHeight || pageHeight <= 0f) break
                if (getPage(1) == null) {
                    scrollY = pageHeight
                    break
                }
                onPageChange?.invoke(1)
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
                    visibilityThreshold = 0.001f
                )
            ) { value, _ ->
                scrollBy(value - lastValue)
                lastValue = value
                invalidate()
            }
        }
    }

    private class ContinuousRenderSnapshot(
        val images: List<Pair<ImagePage, Float>>,
        val screenW: Float,
        val scale: Float,
        val offsetX: Float
    )

    override fun captureRenderState(): Any {
        val screenH = height.toFloat()
        val screenW = width.toFloat()

        var y = synchronized(scrollLock) {
            getPage(0)?.let { page ->
                val pageHeight = getPageHeight(page)
                if (currentPageHeight != pageHeight && scrollY > 0) {
                    currentPageHeight?.let { h -> scrollY -= pageHeight - h }
                    currentPageHeight = pageHeight
                }
            }

            -scrollY
        }

        val images = mutableListOf<Pair<ImagePage, Float>>()

        // Visible band in unscaled page space. Zoom is centered on the screen, so the
        // viewport covers screenH / scale of page space around the screen center.
        val visTop = 0.5f * screenH - screenH / (2f * scale)
        val visBot = 0.5f * screenH + screenH / (2f * scale)

        // Walk forward until the viewport is covered, rather than a fixed page count: zoomed out
        // or with short pages, more of them fit on screen than any constant would allow for, and
        // the ones past it would just be missing.
        var i = 0
        while (y < visBot) {
            val page = getPage(i) ?: break
            val pageHeight = getPageHeight(page)

            if (y + pageHeight > visTop && page.images.any { it != null }) {
                val pageScale = screenW / page.width
                val offsetY = (0.5f * page.height + y / pageScale) / screenH - 0.5f / pageScale
                images.add(Pair(page, offsetY))
            }

            // A zero-height page never advances y, so stop rather than ask for pages forever.
            if (pageHeight <= 0f) break

            y += pageHeight
            i++
        }

        return ContinuousRenderSnapshot(images, screenW, scale, offsetX)
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        tiles.newFrame()
        if (s.images.isEmpty()) return

        // All visible images share one render pass. Pages never overlap vertically, so the clear
        // plus one draw per image writes each pixel once, with no per-image attachment
        // load/store.
        renderPass(encoder, texture) { pass ->
            s.images.forEach { pair ->
                val page = pair.first
                // The snapshot was captured on the main thread; the page can have been evicted
                // since, in which case its image buffers are gone and reading one throws.
                if (!page.destroyed && page.images.any { it != null }) {
                    val pageScale = s.screenW / page.width
                    // Render each image in the page
                    page.images.forEachIndexed { i, image ->
                        if (image != null) {
                            val offsetX = if (page.images.size == 2) {
                                ((0.5f - i) * image.width) / texture.width
                            } else 0f
                            // Sampler shader underneath: several pages can be on screen at once
                            // here, and the view is usually in motion, so the filtered path's
                            // per-pixel cost is not worth paying every frame. The tile cache
                            // draws the filtered version on top as it fills in.
                            val x = s.offsetX / pageScale + offsetX
                            val scale = pageScale * s.scale
                            RenderPage.renderFast(pass, image, texture, x, pair.second, scale)
                            tiles.draw(pass, page, image, texture, x, pair.second, scale)
                        }
                    }
                }
            }
        }
    }
}
