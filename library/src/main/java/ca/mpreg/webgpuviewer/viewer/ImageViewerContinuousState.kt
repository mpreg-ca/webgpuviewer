package ca.mpreg.webgpuviewer.viewer

import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.transition.TransitionBasic

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

    var currentPage: ImagePage? = null
    var currentPageHeight = 0f

    fun scrollBy(deltaPixels: Float) {
        synchronized(mutex) {
            scrollY += deltaPixels

            if (scrollY < 0) {
                if (getPage(-1) == null) {
                    scrollY = 0f
                } else {
                    getPage(0) ?: return
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

    override suspend fun render(encoder: GPUCommandEncoder, texture: GPUTexture) {
        val screenH = height.toFloat()
        val screenW = width.toFloat()

        var y = synchronized(mutex) {
            getPage(0)?.let {
                val pageHeight = getPageHeight(it)
                if (currentPageHeight != pageHeight) {
                    if (currentPage != null) {
                        scrollY -= pageHeight - currentPageHeight
                        currentPage = it
                    }
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

            y += getPageHeight(page)
            if (y >= screenH) break
        }

        images.forEach { pair ->
            pair.first.image?.let {
                val pageScale = screenW / pair.first.width
                TransitionBasic.render(
                    it, encoder, texture, offsetX, pair.second, pageScale * scale
                )
            }
        }
    }
}
