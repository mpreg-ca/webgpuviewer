package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUComputePassEncoder
import androidx.webgpu.GPUQueue
import androidx.webgpu.GPURenderPassEncoder

/*
 * An androidx.webgpu wrapper holds a native reference that only close() drops - nothing releases it
 * when the wrapper is garbage collected. Everything built per frame or per draw has to be closed, or
 * each one leaks a Dawn object (and, for a buffer, the driver memory behind it) for the life of the
 * process. On a phone that grew native memory by tens of megabytes a chapter until the GPU dropped
 * the device.
 *
 * Closing drops only this reference. A pass keeps its own reference to its attachments and to every
 * bind group or vertex buffer set on it, a bind group to what it binds, and a submitted command
 * buffer to all of it - so these are safe as soon as the object is recorded, unlike destroy(), which
 * frees GPU memory at once.
 */

/** Ends the pass, then drops it and anything made only for it, such as its attachment's view. */
internal fun GPURenderPassEncoder.endAndRelease(vararg madeForThisPass: AutoCloseable) {
    end()
    close()
    madeForThisPass.forEach { it.close() }
}

/** Ends the pass, then drops it and anything made only for it. */
internal fun GPUComputePassEncoder.endAndRelease(vararg madeForThisPass: AutoCloseable) {
    end()
    close()
    madeForThisPass.forEach { it.close() }
}

/** Sets a bind group built for this one draw and drops our reference; the pass keeps its own. */
internal fun GPURenderPassEncoder.setTransientBindGroup(index: Int, group: GPUBindGroup) {
    setBindGroup(index, group)
    group.close()
}

/** Sets a bind group built for this one dispatch and drops our reference; the pass keeps its own. */
internal fun GPUComputePassEncoder.setTransientBindGroup(index: Int, group: GPUBindGroup) {
    setBindGroup(index, group)
    group.close()
}

/** Finishes and submits [encoder], then drops it and its command buffer. */
internal fun GPUQueue.submitAndRelease(encoder: GPUCommandEncoder) {
    val commands = encoder.finish()
    submit(arrayOf(commands))
    commands.close()
    encoder.close()
}
