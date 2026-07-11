package ca.mpreg.webgpuviewer.renderer

import android.util.Log
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
    companion object {
        private val device get() = WebGpuRenderer.device
    }

    var textures: MutableList<GPUTexture> = mutableListOf()
    private var tiles: MutableList<GPUTexture> = mutableListOf()

    constructor(pixels: ByteBuffer, width: Int, height: Int, scale: Float, tilesize: Int) : this(
        width = width,
        height = height,
        scale = scale,
        tilesCols = ceil(width.toFloat() / tilesize).toInt(),
        tilesRows = ceil(height.toFloat() / tilesize).toInt(),
        tilesize = tilesize,
    ) {
        for (r in 0 until tilesRows) {
            val tileHeight = min((r + 1) * tilesize, height) - (r * tilesize)
            val y = r * tilesize
            for (c in 0 until tilesCols) {
                val x = c * tilesize
                val tileWidth = min((c + 1) * tilesize, width) - (c * tilesize)

                Log.i("Renderer", "Create tile $c $r $tileWidth $tileHeight $x $y")
                val size = GPUExtent3D(tileWidth, tileHeight)

                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        size = size,
                        format = TextureFormat.RGBA8Unorm,
                        usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment,
                    )
                )

                device.queue.writeTexture(
                    dataLayout = GPUTexelCopyBufferLayout(
                        offset = (y * width + x) * 4L,
                        bytesPerRow = width * Int.SIZE_BYTES,
                        rowsPerImage = height,
                    ),
                    data = pixels,
                    destination = GPUTexelCopyTextureInfo(texture = texture),
                    writeSize = size,
                )

                textures.add(texture)
            }
        }

        for (r in 0 until 2) {
            val row = r.coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + c.coerceAtMost(tilesCols - 1)
                tiles.add(textures[i])
            }
        }
    }

    constructor(texture: GPUTexture, scale: Float, tilesize: Int) : this(
        texture.width, texture.height, scale, 1, 1, tilesize
    ) {
        textures.add(texture)
        repeat(4) {
            tiles.add(texture)
        }
    }

    internal fun cleanup() {
        textures.forEach { tex -> tex.destroy() }
        textures.clear()
    }

    fun update(pixels: ByteBuffer) {
        var i = 0

        for (r in 0 until tilesRows) {
            val tileHeight = min((r + 1) * tilesize, height) - (r * tilesize)
            val y = r * tilesize
            for (c in 0 until tilesCols) {
                val x = c * tilesize
                val tileWidth = min((c + 1) * tilesize, width) - (c * tilesize)

                Log.d("Renderer", "Update tile $c $r")
                val size = GPUExtent3D(tileWidth, tileHeight)

                device.queue.writeTexture(
                    dataLayout = GPUTexelCopyBufferLayout(
                        offset = (y * width + x) * 4L,
                        bytesPerRow = width * Int.SIZE_BYTES,
                        rowsPerImage = height,
                    ),
                    data = pixels,
                    destination = GPUTexelCopyTextureInfo(texture = textures[i++]),
                    writeSize = size,
                )
            }
        }
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
        }.coerceIn(0, tilesCols - 1)

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
                tiles.add(textures[i])
            }
        }

        return Quad(tiles, tX * tiles[0].width, tY * tiles[0].height)
    }
}
