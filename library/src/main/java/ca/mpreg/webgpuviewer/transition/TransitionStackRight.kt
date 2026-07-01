package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage

object TransitionStackRight : Transition() {
    override val code = ""

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        if (frac > 0f) {
            TransitionBasic.render(page1, encoder, dst, 0f, 0f, 1f)
            TransitionBasic.render(page2, encoder, dst, (1f - frac) / page2.scale, 0f, 1f)
        } else {
            TransitionBasic.render(page2, encoder, dst, 0f, 0f, 1f)
            TransitionBasic.render(page1, encoder, dst, -frac / page2.scale, 0f, 1f)
        }
    }
}
