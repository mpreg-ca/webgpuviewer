package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.BlendFactor
import androidx.webgpu.BlendOperation
import androidx.webgpu.GPUBlendComponent
import androidx.webgpu.GPUBlendState
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
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.viewer.ImagePage

abstract class Transition {
    open val code: String = ""

    protected val device get() = WebGpuRenderer.device

    protected open val pipeline: GPURenderPipeline by lazy {
        val shaderModule = device.createShaderModule(
            GPUShaderModuleDescriptor(shaderSourceWGSL = GPUShaderSourceWGSL(code))
        )

        device.createRenderPipeline(
            GPURenderPipelineDescriptor(
                vertex = GPUVertexState(shaderModule, entryPoint = "vs_main"),
                fragment = GPUFragmentState(
                    shaderModule, entryPoint = "fs_main", targets = arrayOf(
                        GPUColorTargetState(
                            format = TextureFormat.RGBA8Unorm, blend = GPUBlendState(
                                color = GPUBlendComponent(
                                    srcFactor = BlendFactor.SrcAlpha,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                ), alpha = GPUBlendComponent(
                                    srcFactor = BlendFactor.One,
                                    dstFactor = BlendFactor.OneMinusSrcAlpha,
                                    operation = BlendOperation.Add
                                )
                            )
                        )
                    )
                ),
                primitive = GPUPrimitiveState(topology = TriangleList),
            )
        )
    }

    abstract fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    )
}