package ca.mpreg.webgpuviewer.transitions

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.WebGpuImageViewerPage

object TransitionStackLeft : Transition() {
    override val code = ""

    override fun render(
        page1: WebGpuImageViewerPage,
        page2: WebGpuImageViewerPage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        if (frac > 0f) {
            TransitionBasic.render(page2, encoder, dst, 0f, 0f, 1f)
            TransitionBasic.render(page1, encoder, dst, -frac / page2.scale, 0f, 1f)
        } else {
            TransitionBasic.render(page1, encoder, dst, 0f, 0f, 1f)
            TransitionBasic.render(page2, encoder, dst, -(frac + 1f) / page2.scale, 0f, 1f)
        }
    }
}
