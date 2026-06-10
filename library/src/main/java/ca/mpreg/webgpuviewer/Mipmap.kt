package ca.mpreg.webgpuviewer

import android.graphics.Bitmap
import android.util.Log
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.min

class Mipmap(
    val width: Int,
    val height: Int,
    val scale: Float,
    val tilesCols: Int,
    val tilesRows: Int,
    val tilesize: Int,
) {
    private var actualTextures: MutableList<GPUTexture> = mutableListOf()
    private var tiles: MutableList<GPUTexture> = mutableListOf()

    var x: Int = 0
    var y: Int = 0

    constructor(
        device: GPUDevice,
        bitmap: Bitmap,
        scale: Float,
        tilesize: Int,
    ) : this(
        width = bitmap.width,
        height = bitmap.height,
        scale = scale,
        tilesCols = ceil(bitmap.width.toFloat() / tilesize).toInt(),
        tilesRows = ceil(bitmap.height.toFloat() / tilesize).toInt(),
        tilesize = tilesize,
    ) {
        ByteBuffer.allocateDirect(bitmap.byteCount).let { pixels ->
            bitmap.copyPixelsToBuffer(pixels)

            for (r in 0 until tilesRows) {
                val height = min((r + 1) * tilesize, height) - (r * tilesize)
                val y = r * tilesize
                for (c in 0 until tilesCols) {
                    val x = c * tilesize
                    val width = min((c + 1) * tilesize, width) - (c * tilesize)

                    Log.i("webgpuviewer", "Create tile " + c + " " + r)
                    val size = GPUExtent3D(width = width, height = height)

                    val texture = device.createTexture(
                        GPUTextureDescriptor(
                            size = size,
                            format = TextureFormat.RGBA8Unorm,
                            usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment,
                        )
                    )

                    device.queue.writeTexture(
                        dataLayout =
                            GPUTexelCopyBufferLayout(
                                offset = (y * bitmap.width + x) * 4L,
                                bytesPerRow = bitmap.width * Int.SIZE_BYTES,
                                rowsPerImage = height,
                            ),
                        data = pixels,
                        destination = GPUTexelCopyTextureInfo(texture = texture),
                        writeSize = size,
                    )

                    actualTextures.add(texture)
                }
            }
        }

        for (r in 0 until 2) {
            val row = r.coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + c.coerceAtMost(tilesCols - 1)
                tiles.add(actualTextures[i])
            }
        }
    }

    fun cleanup() {
        actualTextures.forEach { it.destroy() }
    }

    class Quad(val tiles: List<GPUTexture>, val x: Int, val y: Int)

    fun getQuad(centerX: Int, centerY: Int): Quad {
        if (tilesCols <= 2 && tilesRows <= 2) {
            return Quad(tiles, 0, 0)
        }

        val tiles = mutableListOf<GPUTexture>()

        val cX = centerX.toFloat()
        val cY = centerY.toFloat()

        val c = (cX / tilesize).toInt()
        val tX = when {
            c >= tilesCols - 1 -> tilesCols - 2
            c <= 0 -> 0
            else -> {
                val xCenterRight = if (c + 1 == tilesCols - 1) {
                    ((tilesCols - 1) * tilesize + width) * 0.5
                } else {
                    (c + 1.5) * tilesize
                }

                if (cX - (c - 0.5) * tilesize < xCenterRight - cX) c - 1 else c
            }
        }.coerceIn(0, tilesRows - 1)

        val r = (cY / tilesize).toInt()
        val tY = when {
            r >= tilesRows - 1 -> tilesRows - 2
            r <= 0 -> 0
            else -> {
                val yCenterBottom = if (r + 1 == tilesRows - 1) {
                    ((tilesRows - 1) * tilesize + height) * 0.5
                } else {
                    (r + 1.5) * tilesize
                }

                if (cY - (r - 0.5) * tilesize < yCenterBottom - cY) r - 1 else r
            }
        }.coerceIn(0, tilesRows - 1)

        for (r in 0 until 2) {
            val row = (tY + r).coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + (tX + c).coerceAtMost(tilesCols - 1)
                tiles.add(actualTextures[i])
            }
        }

        return Quad(tiles, x + tX * tiles[0].width, y + tY * tiles[0].height)
    }
}
