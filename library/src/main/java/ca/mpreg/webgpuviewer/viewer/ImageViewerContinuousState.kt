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

    var mutex = Any()
    var scrollY = 0f

    fun getPageHeight(page: ImagePage): Float {
        page.image ?: return page.height.toFloat()
        val fitWidth = width.toFloat() / page.width.toFloat()
        return page.height * fitWidth
    }

    var currentPageHeight: Float? = null

    fun scrollBy(deltaPixels: Float) {
        synchronized(mutex) {
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
                scrollY = 0f
            }
        }
    }

    fun animateScroll(deltaPixels: Float) {
        animationJob?.cancel()
        animationJob = scope?.launch {
            var lastValue = 0f
            animate(
                0f, deltaPixels, animationSpec = spring(stiffness = Spring.StiffnessMediumLow)
            ) { value, _ ->
                scrollBy(value - lastValue)
                lastValue = value
                invalidate()
            }
        }
    }

    override suspend fun render(encoder: GPUCommandEncoder, texture: GPUTexture) {
        val screenH = height.toFloat()
        val screenW = width.toFloat()

        var y = synchronized(mutex) {
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

            page.image?.let {
                images.add(Pair(page, offsetY))
            }

            val pageHeight = getPageHeight(page)

            y += pageHeight
            if (y > screenH / scale + pageHeight) break
        }

        images.forEach { pair ->
            pair.first.image?.let {
                val pageScale = screenW / pair.first.width
                TransitionBasic.render(
                    it, encoder, texture, offsetX / pageScale, pair.second, pageScale * scale
                )
            }
        }
    }
}
