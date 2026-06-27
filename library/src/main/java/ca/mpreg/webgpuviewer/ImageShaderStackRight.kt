package ca.mpreg.webgpuviewer

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture

object ImageShaderStackRight : ImageShader() {
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
        ImageShaderBasic.render(page1, encoder, dst, 0f, 0f, 1f)
        ImageShaderBasic.render(page2, encoder, dst, (1f - frac) / page2.scale, 0f, 1f)
    }
}
