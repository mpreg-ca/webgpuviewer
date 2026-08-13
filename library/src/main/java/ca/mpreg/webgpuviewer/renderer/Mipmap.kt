package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
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
    private var textureViews: MutableList<GPUTextureView> = mutableListOf()
    private var tiles: MutableList<GPUTexture> = mutableListOf()
    private var tileViews: MutableList<GPUTextureView> = mutableListOf()

    private var cachedQuad: Quad? = null

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
                textureViews.add(texture.createView())
            }
        }

        for (r in 0 until 2) {
            val row = r.coerceAtMost(tilesRows - 1) * tilesCols
            for (c in 0 until 2) {
                val i = row + c.coerceAtMost(tilesCols - 1)
                tiles.add(textures[i])
                tileViews.add(textureViews[i])
            }
        }

        if (tilesCols <= 2 && tilesRows <= 2) {
            cachedQuad = Quad(tiles, tileViews, 0, 0)
        }
    }

    constructor(texture: GPUTexture, scale: Float, tilesize: Int) : this(
        texture.width, texture.height, scale, 1, 1, tilesize
    ) {
        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    constructor(width: Int, height: Int) : this(width, height, 1f, 1, 1, 4096) {
        val texture = device.createTexture(
            GPUTextureDescriptor(
                size = GPUExtent3D(width, height),
                format = TextureFormat.RGBA8Unorm,
                usage = TextureUsage.TextureBinding or TextureUsage.CopyDst or TextureUsage.RenderAttachment or TextureUsage.StorageBinding,
            )
        )

        textures.add(texture)
        val view = texture.createView()
        textureViews.add(view)
        repeat(4) {
            tiles.add(texture)
            tileViews.add(view)
        }
        cachedQuad = Quad(tiles, tileViews, 0, 0)
    }

    internal fun cleanup() {
        cachedQuad = null
        lastQuad = null
        lastQuadTX = -1
        lastQuadTY = -1
        textureViews.clear()
        tileViews.clear()
        textures.forEach { tex -> tex.destroy() }
        textures.clear()
        tiles.clear()
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

    class Quad(
        val tiles: List<GPUTexture>, val tileViews: List<GPUTextureView>, val x: Int, val y: Int
    )

    // Cache the last computed quad to avoid allocations when panning within the same tile region
    private var lastQuadTX = -1
    private var lastQuadTY = -1
    private var lastQuad: Quad? = null

    fun getQuad(centerX: Int, centerY: Int): Quad {
        cachedQuad?.let { return it }

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

        // Return cached quad if tile region hasn't changed
        lastQuad?.let { cached ->
            if (lastQuadTX == tX && lastQuadTY == tY) {
                return cached
            }
        }

        // Build a 2x2 quad from the tile grid at (tX, tY) without mutableList overhead
        val r0 = (tY).coerceAtMost(tilesRows - 1) * tilesCols
        val r1 = (tY + 1).coerceAtMost(tilesRows - 1) * tilesCols
        val c0 = (tX).coerceAtMost(tilesCols - 1)
        val c1 = (tX + 1).coerceAtMost(tilesCols - 1)

        val t00 = textures[r0 + c0]
        val v00 = textureViews[r0 + c0]
        val t01 = textures[r0 + c1]
        val v01 = textureViews[r0 + c1]
        val t10 = textures[r1 + c0]
        val v10 = textureViews[r1 + c0]
        val t11 = textures[r1 + c1]
        val v11 = textureViews[r1 + c1]

        val quad = Quad(
            listOf(t00, t01, t10, t11),
            listOf(v00, v01, v10, v11),
            tX * t00.width,
            tY * t00.height
        )
        lastQuadTX = tX
        lastQuadTY = tY
        lastQuad = quad
        return quad
    }
}
