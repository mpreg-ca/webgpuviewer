package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureView

/** What a filter writes. [texture] has `CopySrc`, for copying; [view] is for reading. */
class FilterOutput(val texture: GPUTexture, val view: GPUTextureView)

/**
 * A filter's input on its page. In input pixels: [originX]/[originY], texel (0, 0) from the page's
 * top-left, and [width]/[height], the page. [inputScale] is input pixels per page pixel,
 * [outputSize] the side the result is cut to, [seed] the page's
 * [ca.mpreg.webgpuviewer.viewer.ImagePage.ImageSingle.filterSeed].
 *
 * An [ImageFilterChain]'s page is the image, at scale 1, with seed 0: its level serves every page.
 */
class FilterPlacement(
    val originX: Float,
    val originY: Float,
    val width: Float,
    val height: Float,
    val pageScale: Float,
    val inputScale: Float,
    val outputSize: Float,
    val seed: Int,
) {
    /** For the next filter, whose input is [k] times this one's. */
    fun scaled(k: Float) = if (k == 1f) this else FilterPlacement(
        originX * k, originY * k, width * k, height * k, pageScale, inputScale * k, outputSize, seed
    )
}

/**
 * One step of [ImageFilterChain], run a block at a time as tiles ask; its result is kept as a level
 * of the image. May change resolution by [scale].
 *
 * Behaviour may depend on `pageScale`: [applies] says whether it runs there; [scale] and [halo]
 * are for that scale.
 *
 * Per block, in order, on the render thread: [prepare] binds to the previous step's output and
 * returns its own, then [encode] runs it. [prepare] stays cheap while input texture, size and
 * variant are unchanged.
 */
abstract class ImageFilter {
    protected val device get() = WebGpuRenderer.device

    /** What profiling calls this. */
    open val name: String get() = "filter"

    /** False if this device can't run it; the chain then skips it. */
    open val supported: Boolean get() = true

    /** Set by the holder. Called when output changes for good, e.g. [supported] turning false. */
    @Volatile
    var onUpdate: (() -> Unit)? = null

    open fun applies(pageScale: Float): Boolean = true

    /** Output pixels per input pixel, at [pageScale]. */
    abstract fun scale(pageScale: Float): Float

    /** Output side for a [size]-square input. */
    abstract fun outputSize(size: Int, pageScale: Float): Int

    /** Surrounding input pixels each output pixel needs, cropped from the result; too few and block seams show. */
    open fun halo(pageScale: Float): Int = 0

    /** True when a [size]-square input is within what [prepare] can build. */
    open fun fits(size: Int, pageScale: Float): Boolean = true

    /**
     * Keys the image's cached level: scales with equal signatures must give equal output. Include
     * the running passes when they depend on scale.
     */
    open fun signature(pageScale: Float): String = name

    /** Binds to [input], [size] square; returns what [encode] writes. */
    abstract fun prepare(input: GPUTextureView, size: Int, pageScale: Float): FilterOutput

    /**
     * As [prepare], for a size only being timed: a failure, now or found later, leaves [supported]
     * as it was, and the next [prepare] rebuilds.
     */
    open fun prepareProbe(input: GPUTextureView, size: Int, pageScale: Float): FilterOutput =
        prepare(input, size, pageScale)

    /** Runs on [prepare]'s input, at [placement], in a compute pass writing [timestampWrites]. */
    abstract fun encode(
        encoder: GPUCommandEncoder,
        placement: FilterPlacement,
        timestampWrites: GPUPassTimestampWrites? = null,
    )

    open fun cleanup() {}
}
