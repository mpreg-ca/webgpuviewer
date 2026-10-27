package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.GPUCommandEncoder
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.submitAndRelease

object Draw {
    internal val device get() = WebGpuRenderer.device

    fun submit(block: Draw.(GPUCommandEncoder) -> Unit) {
        var encoder: GPUCommandEncoder? = device.createCommandEncoder()
        try {
            block.invoke(this, encoder!!)
            device.queue.submitAndRelease(encoder)
            encoder = null
        } finally {
            encoder?.close()
        }
    }
}
