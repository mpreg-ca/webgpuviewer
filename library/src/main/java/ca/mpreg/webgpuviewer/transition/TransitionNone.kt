package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPUTexture
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import ca.mpreg.webgpuviewer.renderer.TileRenderer
import ca.mpreg.webgpuviewer.transition.Transition.Companion.renderCacheSeed
import ca.mpreg.webgpuviewer.viewer.ImagePage

/**
 * No animation: [page1] stays on screen for the whole drag and the turn just jumps once it
 * commits (handled outside [render] - by the time [page2] would show, offset is back to 0 and
 * the normal live-render path takes over). [frac]/[page2] are unused since there's nothing to
 * blend or slide toward. Draws [page1] straight to [dst] via [renderCacheSeed], skipping the
 * getCachedTexture/blitCached cache-texture indirection every other transition uses.
 */
object TransitionNone : Transition() {
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
        val pass = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = dst.createView(),
                        loadOp = LoadOp.Clear,
                        storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )
        try {
            renderCacheSeed(pass, page1, dst, tiles)
        } finally {
            pass.end()
        }
    }
}
