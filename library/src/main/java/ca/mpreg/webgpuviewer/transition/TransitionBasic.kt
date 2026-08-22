package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.draw.Draw
import ca.mpreg.webgpuviewer.draw.clear
import ca.mpreg.webgpuviewer.draw.rect
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.transition.Transition.Companion.getCachedTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage

/**
 * Background behind [page]'s slide position, spanning the whole (screen-shaped) cached surface at
 * normalised offset ([offsetX], [offsetY]) - same convention [blitCached] blits that surface at.
 * As [TransitionCube.drawFace]'s own background column: a flat rect per image from that image's
 * own [ca.mpreg.webgpuviewer.renderer.Image.backgroundColor], not [RenderPage.renderBackground]'s
 * fades, since the cached surface (not just the image) is what's being slid into place here. Drawn
 * per image (own column for LEFT/RIGHT, matching [RenderPage]) rather than one flat full-width
 * rect - a spread with only one side present must not have its background spill into the empty
 * other side.
 */
internal fun drawBackground(
    encoder: GPUCommandEncoder, dst: GPUTexture, page: ImagePage, offsetX: Float, offsetY: Float
) {
    val renderImages = if (page.images.size == 1) listOf(page.image) else page.images
    renderImages.forEach { image ->
        image ?: return@forEach
        if (image.mipmaps.isEmpty()) return@forEach
        val imgOffsetX = when (image.position) {
            Image.Position.LEFT -> (-0.5f * image.width) / dst.width
            Image.Position.RIGHT -> (0.5f * image.width) / dst.width
            Image.Position.SINGLE -> 0f
        }
        val rect = image.placement(dst, page.x + imgOffsetX, page.y, page.scale)
        val x1 = if (image.position == Image.Position.SINGLE) 0f else rect[0]
        val x2 = if (image.position == Image.Position.SINGLE) 1f else rect[2]
        Draw.rect(
            encoder,
            dst,
            offsetX + x1,
            offsetY,
            offsetX + x2,
            offsetY + 1f,
            image.backgroundColor
        )
    }
}

/**
 * Slide transition: both pages go to cached textures, then get blitted side by side at an offset.
 *
 * Drawing a page is [TileRenderer.renderFullyTiled]'s job; this only decides where they end up.
 *
 * [getCachedTexture] keys on the page's own transform, and a page turn only animates the offset,
 * so that render happens once per transition and every later frame is a cache hit plus a 1:1 blit.
 * Each page's background is drawn separately, live, at the same offset - see [drawBackground].
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
        tiles: TileRenderer,
    ) {
        val cached1 = getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

        val cached2 = getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

        Draw.clear(encoder, dst, 0)

        if (frac > 0f) {
            drawBackground(encoder, dst, page2, 1f - frac, 0f)
            drawBackground(encoder, dst, page1, -frac, 0f)
            blitCached(encoder, dst, cached2, 1f - frac, 0f)
            blitCached(encoder, dst, cached1, -frac, 0f)
        } else {
            drawBackground(encoder, dst, page2, -(frac + 1f), 0f)
            drawBackground(encoder, dst, page1, -frac, 0f)
            blitCached(encoder, dst, cached2, -(frac + 1f), 0f)
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
            tiles: TileRenderer,
        ) {
            val cached1 =
                getCachedTexture(page1, true, encoder, dst.width, dst.height, tiles)

            val cached2 =
                getCachedTexture(page2, false, encoder, dst.width, dst.height, tiles)

            Draw.clear(encoder, dst, 0)

            if (frac > 0f) {
                drawBackground(encoder, dst, page1, 0f, -frac)
                drawBackground(encoder, dst, page2, 0f, 1f - frac)
                blitCached(encoder, dst, cached1, 0f, -frac)
                blitCached(encoder, dst, cached2, 0f, 1f - frac)
            } else {
                drawBackground(encoder, dst, page2, 0f, -(frac + 1f))
                drawBackground(encoder, dst, page1, 0f, -frac)
                blitCached(encoder, dst, cached2, 0f, -(frac + 1f))
                blitCached(encoder, dst, cached1, 0f, -frac)
            }
        }
    }
}
