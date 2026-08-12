package ca.mpreg.webgpuviewer.viewer

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.transition.TransitionBasic
import kotlinx.coroutines.launch

class ImageViewerContinuousState : ImageViewerState(isVertical = true) {
    var scale = 1f

    var offsetX = 0f

    private val scrollLock = Any()
    var scrollY = 0f

    fun getPageHeight(page: ImagePage): Float {
        if (page.images.all { it == null }) return page.height.toFloat()
        val fitWidth = width.toFloat() / page.width.toFloat()
        return page.height * fitWidth
    }

    var currentPageHeight: Float? = null

    fun scrollBy(deltaPixels: Float) {
        synchronized(scrollLock) {
            getPage(0) ?: return

            scrollY += deltaPixels

            if (scrollY < 0) {
                if (getPage(-1) == null) {
                    scrollY = 0f
                } else {
                    onPageChange?.invoke(-1)
                    val newPage = getPage(0) ?: return
                    val pageHeight = getPageHeight(newPage)
                    currentPageHeight = pageHeight
                    scrollY += pageHeight
                }
            }

            val page = getPage(0) ?: return
            val pageHeight = getPageHeight(page)
            if (scrollY > pageHeight) {
                onPageChange?.invoke(1)
                val newPage = getPage(0) ?: return
                currentPageHeight = getPageHeight(newPage)
                scrollY -= pageHeight
            }

            if (getPage(1) == null) {
                val page = getPage(0) ?: return
                val pageHeight = getPageHeight(page)
                scrollY = scrollY.coerceAtMost(pageHeight)
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

        for (i in 0 until 6) {
            val page = getPage(i) ?: break
            val pageScale = screenW / page.width
            val offsetY = (0.5f * page.height + y / pageScale) / screenH - 0.5f / pageScale

            if (page.images.any { it != null }) {
                images.add(Pair(page, offsetY))
            }

            val pageHeight = getPageHeight(page)

            y += pageHeight
            if (y > screenH / scale + pageHeight) break
        }

        return ContinuousRenderSnapshot(images, screenW, scale, offsetX)
    }

    override suspend fun renderSnapshot(
        encoder: GPUCommandEncoder,
        texture: GPUTexture,
        snapshot: Any
    ) {
        val s = snapshot as ContinuousRenderSnapshot
        s.images.forEach { pair ->
            val page = pair.first
            if (page.images.any { it != null }) {
                val pageScale = s.screenW / page.width
                // Render each image in the page
                page.images.forEachIndexed { i, image ->
                    if (image != null) {
                        val offsetX = if (page.images.size == 2) {
                            ((0.5f - i) * image.width) / texture.width
                        } else 0f
                        TransitionBasic.render(
                            image,
                            encoder,
                            texture,
                            s.offsetX / pageScale + offsetX,
                            pair.second,
                            pageScale * s.scale
                        )
                    }
                }
            }
        }
    }
}
