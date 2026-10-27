package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPUQuerySet
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureUsage
import kotlin.math.ceil

/**
 * A profiled tile's timestamp queries, in [TileRenderer]'s query set: 0 tile start, 1 tile end, 2 page
 * draw end, then a begin/end pair from 3 for each of the first [PROFILED_FILTERS] filters, and the final
 * resize's start last. Later filters go unmeasured, as untimed gaps.
 */
internal const val PROFILED_FILTERS = 5

/** Query indices for the [i]th running filter's pass. */
internal fun filterQueries(i: Int): Pair<Int, Int> = 3 + 2 * i to 4 + 2 * i

internal const val RESIZE_BEGIN_QUERY = 3 + 2 * PROFILED_FILTERS

internal const val PROFILE_QUERY_COUNT = RESIZE_BEGIN_QUERY + 1

/**
 * Draws a high-quality tile into the atlas, from [ImageFilterChain]'s filtered level if any, else the
 * image's own levels.
 *
 * Without filters (the default) a tile resolves in one step, through [upscaler] or [downscaler]. With
 * them - the same [ImageFilter]s an [ImageFilterChain] runs, but per tile and living only as long as it -
 * the source is drawn at its own resolution into [input] with the chain's halo, each filter applying at
 * the tile's page scale runs on the previous one's output, and [resolve] resizes the last to the tile
 * (box average to shrink, Catmull-Rom to enlarge).
 *
 * Held by [TileRenderer]; everything but the samplers on the render thread. Changing a sampler, or a
 * filter failing, calls [onChange]: tiles cut before are wrong.
 */
internal class TilesFilterChain(private val onChange: () -> Unit) {
    private val device get() = WebGpuRenderer.device

    // ---- the one-step resolve ----

    /** Used where a tile enlarges in one step. Safe from any thread. */
    @Volatile
    var upscaler: Sampler = Sampler.catmullRom()
        set(value) {
            if (field === value) return
            field = value
            onChange()
        }

    /** Used where a tile shrinks in one step. Safe from any thread. */
    @Volatile
    var downscaler: Sampler = Sampler.box()
        set(value) {
            if (field === value) return
            field = value
            onChange()
        }

    // Re-derived only when a sampler is swapped, by identity: a sampler's code is fixed.
    private var filteredCache: RenderPage.Filtered? = null
    private var filteredUp: Sampler? = null
    private var filteredDown: Sampler? = null

    /** The current samplers' resolves - see [Sampler.code]. */
    fun filtered(): RenderPage.Filtered {
        val up = upscaler
        val down = downscaler
        var pair = filteredCache
        if (pair == null || up !== filteredUp || down !== filteredDown) {
            pair = RenderPage.filtered(up.code, down.code)
            filteredCache = pair
            filteredUp = up
            filteredDown = down
        }
        return pair
    }

    /**
     * Draw [image] through the samplers, placed as for [RenderPage.render], from [source] (its filtered
     * level) if given, else its own levels; [levelZero] forces the base one.
     */
    fun draw(
        pass: GPURenderPassEncoder,
        image: Image,
        target: GPUTexture,
        x: Float,
        y: Float,
        scale: Float,
        source: Mipmap?,
        levelZero: Boolean,
        extend: Extend? = null,
    ) = RenderPage.render(
        pass, image, target, x, y, scale, filtered(), extend, levelZero, source
    )

    // ---- the filters ----

    /** Run on each tile, in order. Assign to change. */
    var filters: List<ImageFilter> = emptyList()
        set(value) {
            field.forEach { it.onUpdate = null }
            field = value.toList()
            field.forEach { it.onUpdate = changed }
            prepared = null
            broken = false
        }

    private val changed: () -> Unit = {
        prepared = null
        onChange()
    }

    /**
     * Set when [input] fails, until [filters] is reassigned: otherwise the plain-tile upgrade would
     * re-cut the tiles that fell back every frame.
     */
    private var broken = false

    val isActive: Boolean get() = !broken && filters.any { it.supported }

    private fun running(pageScale: Float): List<ImageFilter> =
        if (broken) emptyList() else filters.filter { it.supported && it.applies(pageScale) }

    fun runsAt(pageScale: Float) = running(pageScale).isNotEmpty()

    fun names(pageScale: Float) = running(pageScale).map { it.name }

    /** Output pixels per input pixel, over all running filters. */
    private fun scale(pageScale: Float) =
        running(pageScale).fold(1f) { product, f -> product * f.scale(pageScale) }

    /**
     * Source pixels the filters need around the tile: from the last back, each one's halo plus the
     * later ones' need, in its own input pixels.
     */
    fun halo(pageScale: Float): Int {
        var margin = 0
        for (f in running(pageScale).asReversed()) {
            margin = f.halo(pageScale) + ceil(margin / f.scale(pageScale)).toInt()
        }
        return margin
    }

    /**
     * Source pixels per side the tile covers at [scale] tile pixels per source pixel, plus one for a
     * start between pixels. Excludes [halo].
     */
    fun inputSpan(tileSize: Int, scale: Float): Int = ceil(tileSize / scale).toInt() + 1

    /** True when the filters can be built for this tile. [scale] is tile pixels per source pixel. */
    fun fits(tileSize: Int, scale: Float, pageScale: Float): Boolean {
        var size = inputSpan(tileSize, scale) + 2 * halo(pageScale)
        if (size > WebGpuRenderer.maxTextureDimension2D) return false
        return try {
            for (f in running(pageScale)) {
                if (!f.fits(size, pageScale)) return false
                size = f.outputSize(size, pageScale)
            }
            true
        } catch (e: IllegalArgumentException) {
            false
        }
    }

    // Objects of a device that was lost and replaced are dropped, not closed: calling into them
    // crashed in native code.
    private var builtOn: GPUDevice? = null
    private var inputTexture: GPUTexture? = null
    private var inputView: GPUTextureView? = null
    private var inputSize = 0
    private var inputFormat = 0

    /** What [resolve] reads, from the last [input]. Filters re-prepare every tile. */
    private var prepared: FilterOutput? = null

    /**
     * The `size`-square colour target (hence [format]) the tile's source is drawn into, with each filter
     * bound to the previous one's output. Null if a filter can't run: the tile then resolves in one step.
     */
    fun input(size: Int, format: Int, pageScale: Float): GPUTexture? {
        prepared = null
        return try {
            if (size < 1 || size > WebGpuRenderer.maxTextureDimension2D) return null
            val device = device
            if (builtOn != null && builtOn !== device) {
                inputTexture = null
                inputView = null
                resize.forget()
            }
            builtOn = device

            var texture = inputTexture
            if (texture == null || inputSize != size || inputFormat != format) {
                inputView?.close()
                texture?.destroyAndRelease()
                inputTexture = null
                inputView = null
                texture = device.createTexture(
                    GPUTextureDescriptor(
                        label = "tiles filter input",
                        size = GPUExtent3D(size, size),
                        format = format,
                        usage = TextureUsage.RenderAttachment or TextureUsage.TextureBinding,
                    )
                )
                inputTexture = texture
                inputView = texture.createView()
                inputSize = size
                inputFormat = format
            }

            var view = inputView!!
            var side = size
            var out: FilterOutput? = null
            for (f in running(pageScale)) {
                out = f.prepare(view, side, pageScale)
                view = out.view
                side = f.outputSize(side, pageScale)
            }
            prepared = out
            if (out != null) texture else null
        } catch (e: Exception) {
            Log.w(TAG, "A filter could not be prepared, resolving in one step", e)
            broken = true
            null
        }
    }

    /** [input]'s view, not remade per tile. Valid only after [input] returns. */
    val view: GPUTextureView? get() = inputView

    /**
     * Encode the running filters over [input]'s current contents. With [queries], the first
     * [PROFILED_FILTERS] write the timestamps [filterQueries] gives.
     */
    fun encode(encoder: GPUCommandEncoder, pageScale: Float, queries: GPUQuerySet? = null) {
        running(pageScale).forEachIndexed { i, f ->
            val (begin, end) = filterQueries(i)
            f.encode(
                encoder,
                if (queries != null && i < PROFILED_FILTERS) GPUPassTimestampWrites(
                    queries,
                    begin,
                    end
                ) else null
            )
        }
    }

    private val resize = TileResize()

    /**
     * Draw the tile from the filters' result. [scale] is tile pixels per source pixel (the source was
     * drawn at 1), [pageScale] what the filters ran for, [fracX]/[fracY] the tile's start within the
     * first source pixel drawn.
     */
    fun resolve(
        pass: GPURenderPassEncoder,
        format: Int,
        scale: Float,
        pageScale: Float,
        fracX: Float,
        fracY: Float,
    ) {
        val out = prepared ?: return
        val k = this.scale(pageScale)
        val halo = halo(pageScale)
        resize.draw(pass, format, out.view, k * (halo + fracX), k * (halo + fracY), k / scale)
    }

    /** Rebuilt on the next [input]. */
    fun cleanup() {
        filters.forEach { it.cleanup() }
        inputView?.close()
        inputTexture?.destroyAndRelease()
        inputTexture = null
        inputView = null
        prepared = null
        resize.cleanup()
    }

    private companion object {
        const val TAG = "TilesFilterChain"
    }
}
