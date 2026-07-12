package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoder
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer

object Draw {
    internal val device get() = WebGpuRenderer.device

    private val tempBuffers = mutableListOf<GPUBuffer>()

    fun submit(block: Draw.(GPUCommandEncoder) -> Unit) {
        val encoder = device.createCommandEncoder()
        block.invoke(this, encoder)
        device.queue.submit(arrayOf(encoder.finish()))
        tempBuffers.forEach { it.destroy() }
        tempBuffers.clear()
    }

    internal fun createBuffer(size: Long, usage: Int): GPUBuffer {
        return device.createBuffer(GPUBufferDescriptor(size = size, usage = usage)).also {
            tempBuffers.add(it)
        }
    }
}
