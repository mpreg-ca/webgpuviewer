package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.RenderPage
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage

/**
 * Slide transition: both pages go to cached textures, then get blitted side by side at an offset.
 *
 * Drawing a page is [RenderPage]'s job; this only decides where the two end up.
 *
 * The page goes through [RenderPage.render] - the sharp, expensive filter - rather than
 * [RenderPage.renderFast]. [getCachedTexture] keys on the page's own transform, and a page turn
 * only animates the offset, so that render happens once per transition and every later frame is a
 * cache hit plus a 1:1 blit. Paying for the good filter once is free; using the cheap one would
 * bake softness into every frame of the animation.
 */
object TransitionBasic : Transition() {
    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height) { pass, tex ->
            RenderPage.render(pass, page1, tex, 0f, 0f, 1f)
        }

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height) { pass, tex ->
            RenderPage.render(pass, page2, tex, 0f, 0f, 1f)
        }

        if (frac > 0f) {
            blitCached(encoder, dst, cached2, 1f - frac, 0f, clearFirst = true)
            blitCached(encoder, dst, cached1, -frac, 0f)
        } else {
            blitCached(encoder, dst, cached2, -(frac + 1f), 0f, clearFirst = true)
            blitCached(encoder, dst, cached1, -frac, 0f)
        }
    }

    /** The same slide, vertically. */
    object Vertical : Transition() {
        override fun render(
            page1: ImagePage,
            page2: ImagePage,
            encoder: GPUCommandEncoder,
            dst: GPUTexture,
            frac: Float,
            pos1: Offset,
            pos2: Offset,
        ) {
            val cached1 =
                getCachedTexture(page1, true, encoder, dst.width, dst.height) { pass, tex ->
                    RenderPage.render(pass, page1, tex, 0f, 0f, 1f)
                }

            val cached2 =
                getCachedTexture(page2, false, encoder, dst.width, dst.height) { pass, tex ->
                    RenderPage.render(pass, page2, tex, 0f, 0f, 1f)
                }

            if (frac > 0f) {
                blitCached(encoder, dst, cached1, 0f, -frac, clearFirst = true)
                blitCached(encoder, dst, cached2, 0f, 1f - frac)
            } else {
                blitCached(encoder, dst, cached2, 0f, -(frac + 1f), clearFirst = true)
                blitCached(encoder, dst, cached1, 0f, -frac)
            }
        }
    }
}
