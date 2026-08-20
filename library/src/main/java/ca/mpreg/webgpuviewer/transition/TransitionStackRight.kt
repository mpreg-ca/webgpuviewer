package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage

object TransitionStackRight : Transition() {
    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
        tiles: TileRenderer,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height) { pass, tex ->
            renderForCache(pass, page1, tex, tiles)
        }

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height) { pass, tex ->
            renderForCache(pass, page2, tex, tiles)
        }

        Draw.clear(encoder, dst, 0)

        if (frac > 0f) {
            drawBackground(encoder, dst, page1, 0f, 0f)
            blitCached(encoder, dst, cached1, 0f, 0f)
            drawBackground(encoder, dst, page2, 1f - frac, 0f)
            blitCached(encoder, dst, cached2, 1f - frac, 0f)
        } else {
            drawBackground(encoder, dst, page2, 0f, 0f)
            blitCached(encoder, dst, cached2, 0f, 0f)
            drawBackground(encoder, dst, page1, -frac, 0f)
            blitCached(encoder, dst, cached1, -frac, 0f)
        }
    }
}
