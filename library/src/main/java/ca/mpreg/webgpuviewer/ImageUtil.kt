package ca.mpreg.webgpuviewer

import java.nio.ByteBuffer

object ImageUtil {
    init {
        System.loadLibrary("resize")
    }

    external fun resizeLinearAreaNative(
        pixels: ByteBuffer,
        dstPixels: ByteBuffer,
        width: Int,
        height: Int
    )

    fun resize(source: ByteBuffer, width: Int, height: Int): ByteBuffer {
        val output = ByteBuffer.allocateDirect(width * height)
        resizeLinearAreaNative(source, output, width, height)
        return output
    }
}
