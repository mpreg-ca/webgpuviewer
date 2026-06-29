package ca.mpreg.webgpuviewer.transitions

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUColorTargetState
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUFragmentState
import androidx.webgpu.GPUPrimitiveState
import androidx.webgpu.GPURenderPipeline
import androidx.webgpu.GPURenderPipelineDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUVertexState
import androidx.webgpu.PrimitiveTopology.Companion.TriangleList
import androidx.webgpu.TextureFormat
import ca.mpreg.webgpuviewer.WebGpuImageViewerPage
import ca.mpreg.webgpuviewer.WebGpuRenderer

abstract class Transition {
    protected abstract val code: String

    protected val pipeline: GPURenderPipeline by lazy {
        val shaderModule = WebGpuRenderer.device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        WebGpuRenderer.device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule,
                    entryPoint = "fs_main",
                    targets = arrayOf(GPUColorTargetState(format = TextureFormat.RGBA8Unorm))
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    abstract fun render(
        page1: WebGpuImageViewerPage,
        page2: WebGpuImageViewerPage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    )
}