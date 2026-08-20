package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage

object TransitionStackDown : Transition() {
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
            tiles.renderFullyTiled(pass, page1, tex)
        }

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height) { pass, tex ->
            tiles.renderFullyTiled(pass, page2, tex)
        }

        if (frac > 0f) {
            blitCached(encoder, dst, cached2, 0f, 0f, clearFirst = true)
            blitCached(encoder, dst, cached1, 0f, frac)
        } else {
            blitCached(encoder, dst, cached1, 0f, 0f, clearFirst = true)
            blitCached(encoder, dst, cached2, 0f, frac + 1f)
        }
    }
}
