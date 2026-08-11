package ca.mpreg.webgpuviewer.draw

import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil

private val device get() = WebGpuRenderer.device

private val pipeline: GPUComputePipeline by lazy {
    device.createComputePipeline(
        GPUComputePipelineDescriptor(
            GPUComputeState(
                device.createShaderModule(
                    GPUShaderModuleDescriptor(
                        shaderSourceWGSL = GPUShaderSourceWGSL(RECT_SHADER)
                    )
                )
            )
        )
    )
}

private const val RECT_SHADER = """
struct Params {
    rect: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_tex);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let x = f32(id.x);
    let y = f32(id.y);
    
    let left = params.rect.x;
    let top = params.rect.y;
    let right = params.rect.z;
    let bottom = params.rect.w;
    
    if (x >= left && x < right && y >= top && y < bottom) {
        textureStore(output_tex, vec2<i32>(id.xy), params.color);
    }
}
"""

// Thread-local ByteBuffer to avoid allocation per call
private val byteBufferLocal = ThreadLocal.withInitial {
    ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder())
}

/**
 * Draw a filled rectangle with the specified color.
 * Coordinates are in normalized [0, 1] range.
 */
fun Draw.rect(
    encoder: GPUCommandEncoder,
    texture: GPUTexture,
    x1: Float,
    y1: Float,
    x2: Float,
    y2: Float,
    color: Int
) {
    val r = ((color shr 16) and 0xFF) / 255f
    val g = ((color shr 8) and 0xFF) / 255f
    val b = (color and 0xFF) / 255f
    val a = ((color ushr 24) and 0xFF) / 255f

    val left = x1 * texture.width
    val top = y1 * texture.height
    val right = x2 * texture.width
    val bottom = y2 * texture.height

    val byteBuffer = byteBufferLocal.get()
    byteBuffer.clear()
    byteBuffer.putFloat(left)
    byteBuffer.putFloat(top)
    byteBuffer.putFloat(right)
    byteBuffer.putFloat(bottom)
    byteBuffer.putFloat(r)
    byteBuffer.putFloat(g)
    byteBuffer.putFloat(b)
    byteBuffer.putFloat(a)
    byteBuffer.flip()

    // Buffer is created per-call because it must persist until GPU work completes.
    // WebGPU drivers efficiently pool small uniform buffers.
    val uniformBuffer = device.createBuffer(
        GPUBufferDescriptor(size = 32, usage = BufferUsage.Uniform or BufferUsage.CopyDst)
    )
    device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

    val dispatchW = ceil(texture.width / 8f).toInt()
    val dispatchH = ceil(texture.height / 8f).toInt()

    val pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(
        0, device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                    GPUBindGroupEntry(0, textureView = texture.createView()),
                    GPUBindGroupEntry(1, buffer = uniformBuffer),
                )
            )
        )
    )
    pass.dispatchWorkgroups(dispatchW, dispatchH)
    pass.end()
}
