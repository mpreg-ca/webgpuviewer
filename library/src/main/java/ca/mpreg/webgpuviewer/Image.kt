package ca.mpreg.webgpuviewer

import android.graphics.Bitmap
import androidx.compose.ui.unit.IntOffset
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUDevice
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Image(val width: Int, val height: Int) {
    var position: IntOffset = IntOffset.Zero
    var scale: Float = 1f

    var mipmaps: MutableList<Mipmap> = mutableListOf()

    var byteBuffer: ByteBuffer = ByteBuffer.allocateDirect(32)
    lateinit var buffer: GPUBuffer

    constructor(
        device: GPUDevice,
        bitmap: Bitmap,
    ) : this(width = bitmap.width, height = bitmap.height) {
        byteBuffer.order(ByteOrder.nativeOrder())

        buffer = webgpu.device!!.createBuffer(
            GPUBufferDescriptor(
                size = 32,
                usage = BufferUsage.CopyDst or BufferUsage.Uniform
            )
        )

        mipmaps.add(Mipmap(device, bitmap, 1f, 4096))
    }

    fun cleanup() {
        mipmaps.forEach { it.cleanup() }
        buffer.close()
    }
}
