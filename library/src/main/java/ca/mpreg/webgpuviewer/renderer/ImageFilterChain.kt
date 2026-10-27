package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUColor
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPURenderPassColorAttachment
import androidx.webgpu.GPURenderPassDescriptor
import androidx.webgpu.GPURenderPassEncoder
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.LoadOp
import androidx.webgpu.StoreOp
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.round

/**
 * Starting output block side, in image pixels: a block, once made, serves every scale and tile.
 * [ImageFilterChain.blockSize] moves among [BLOCK_SIZES].
 */
const val BLOCK_SIZE = 256

/**
 * Smallest first; powers of two, so each divides a level tile cut for another. Capped at [BLOCK_SIZE]:
 * filter textures grow with the block, and running out of memory disables a filter for good.
 */
internal val BLOCK_SIZES = intArrayOf(64, 128, BLOCK_SIZE)

/** In image pixels. */
internal class ImageRect(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * Copies one block's output, halo cropped, into the level at the pass's viewport. Unpremultiplies: the
 * network runs premultiplied, levels hold straight alpha. Colour is clamped to [max], 1 for an 8-bit
 * level and a half float's largest for an HDR one.
 */
private fun write(max: String) = """
struct Params {
    origin: vec2<i32>,
    inset: vec2<i32>,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(floor(in.position.xy)) - params.origin + params.inset;
    let c = clamp(textureLoad(src, p, 0), vec4<f32>(0.0), vec4<f32>($max, $max, $max, 1.0));
    let rgb = select(vec3<f32>(0.0), c.rgb / max(c.a, 0.0001), c.a > 0.0);
    return vec4<f32>(clamp(rgb, vec3<f32>(0.0), vec3<f32>($max)), c.a);
}
"""

/**
 * All levels together: 96MB per GB of the device's memory (to the nearest power of two, 0.25 to 8, as a
 * browser reports it), max 512MB; else 256MB.
 */
private val BUDGET_BYTES: Long by lazy {
    val kb = try {
        File("/proc/meminfo").useLines { lines ->
            lines.firstOrNull { it.startsWith("MemTotal:") }
                ?.split(Regex("""\s+"""))?.getOrNull(1)?.toLongOrNull()
        }
    } catch (e: Exception) {
        null
    }
    val megabytes = if (kb == null || kb <= 0) 256.0 else {
        val gb = 2.0.pow(round(log2(kb / 1048576.0))).coerceIn(0.25, 8.0)
        min(512.0, gb * 96)
    }
    (megabytes * (1 shl 20)).toLong()
}

/** Max level tile side, in texels: a level's GPU memory is allocated and freed per tile. */
private const val MAX_LEVEL_TILE = 4096

/** One image's filtered level and which of its blocks are made. */
private class Level(
    val image: Image,
    val signature: String,
    val version: Int,
    val k: Int,
    val blockSize: Int,
    val mipmap: Mipmap,
    val columns: Int,
    val rows: Int,
) {
    /** 1 where a block is made, by `row * columns + column`. */
    val filled = BooleanArray(columns * rows)

    /** Tiles with a texture, by `row * tilesCols + column`. */
    val slots = HashMap<Int, Slot>()

    /** Blocks per tile side. */
    val blocksPerTile: Int get() = mipmap.tilesize / (k * blockSize)
}

/** One allocated tile of a level. */
private class Slot(val level: Level, val column: Int, val row: Int, val bytes: Long) {
    /** [TileRenderer]'s frame it was last read in. */
    var used = -1L
}

internal class BlockGeometry(
    /** Output texels per image pixel. */
    val k: Int,
    /** Image pixels read around a block. */
    val halo: Int,
    /** Input block side with halo, in pixels. */
    val side: Int,
    /** Output block side, in image pixels. */
    val blockSize: Int,
)

/**
 * Image filters. Their output is kept as another level of the image, [k] times its size, filled a block
 * at a time as tiles ask: the chain run on that block plus its halo. A block is made once for every
 * scale; tiles read the level like any other, via [TilesFilterChain]'s resolve.
 *
 * Level tiles get textures when first wanted ([reserve]) and are evicted LRU within [BUDGET_BYTES],
 * never one read this frame. A level is dropped when its image's content, the running filters, or its
 * page change ([forget]). A level holds its image's format, so an HDR page stays HDR.
 *
 * Held by [TileRenderer], render thread only; filters are `ImageViewerState.imageFilters`.
 */
internal class ImageFilterChain(private val onChange: () -> Unit) {
    private val device get() = WebGpuRenderer.device

    /**
     * One of [BLOCK_SIZES], picked by [TileRenderer] from measured cost. A level at another size is
     * dropped when next asked for ([level]); cut tiles keep their pixels.
     */
    var blockSize = BLOCK_SIZE

    /** In run order. Assign to change. */
    var filters: List<ImageFilter> = emptyList()
        set(value) {
            field.forEach { it.onUpdate = null }
            field = value.toList()
            field.forEach { it.onUpdate = changed }
            prepared = null
            broken = false
            clear()
        }

    private val changed: () -> Unit = {
        prepared = null
        clear()
        onChange()
    }

    /** Set when the chain fails to build; tiles draw from the image until [filters] is reassigned. */
    private var broken = false

    val isActive: Boolean get() = !broken && filters.any { it.supported }

    private fun running(pageScale: Float): List<ImageFilter> =
        if (broken) emptyList() else filters.filter { it.supported && it.applies(pageScale) }

    fun runsAt(pageScale: Float) = running(pageScale).isNotEmpty()

    fun names(pageScale: Float) = running(pageScale).map { it.name }

    /** Keys what the chain made at [pageScale]: the running filters, in order, and each one's passes. */
    fun signature(pageScale: Float) =
        running(pageScale).joinToString("|") { it.signature(pageScale) }

    private fun scale(pageScale: Float) =
        running(pageScale).fold(1f) { product, f -> product * f.scale(pageScale) }

    /**
     * Image pixels needed around a block at [pageScale], from the last filter back: each one's
     * halo plus what later ones need, in its input pixels.
     */
    private fun halo(pageScale: Float): Int {
        var margin = 0
        for (f in running(pageScale).asReversed()) {
            margin = f.halo(pageScale) + ceil(margin / f.scale(pageScale)).toInt()
        }
        return margin
    }

    private fun fitsInput(size: Int, pageScale: Float): Boolean {
        if (size > WebGpuRenderer.maxTextureDimension2D) return false
        var side = size
        return try {
            for (f in running(pageScale)) {
                if (!f.fits(side, pageScale)) return false
                side = f.outputSize(side, pageScale)
            }
            true
        } catch (e: IllegalArgumentException) {
            false
        }
    }

    private fun outputSide(size: Int, pageScale: Float) =
        running(pageScale).fold(size) { side, f -> f.outputSize(side, pageScale) }

    /**
     * Null where nothing runs. Throws if the chain can't run on [blockSize] blocks: such filters go in
     * [TilesFilterChain].
     */
    fun geometry(pageScale: Float, blockSize: Int = this.blockSize): BlockGeometry? {
        if (!runsAt(pageScale)) return null
        val names = names(pageScale).joinToString(", ")
        val scale = scale(pageScale)
        val k = scale.toInt()
        check(k >= 1 && k.toFloat() == scale) { "ImageFilterChain [$names]: scale $scale is not a whole number" }
        val halo = halo(pageScale)
        val side = blockSize + 2 * halo
        check(
            fitsInput(
                side,
                pageScale
            )
        ) { "ImageFilterChain [$names]: cannot run on a ${side}px block" }
        val out = outputSide(side, pageScale)
        check(out == k * side) { "ImageFilterChain [$names]: a ${side}px block gives ${out}px, not ${k * side}px" }
        return BlockGeometry(k, halo, side, blockSize)
    }

    private val levels = HashMap<Image, Level>()

    /** All levels' allocated tiles, LRU first. */
    private val slots = LinkedHashSet<Slot>()
    private var totalBytes = 0L

    /** Lowered after the device runs out of memory. */
    private var cap = Long.MAX_VALUE

    private val budget: Long get() = min(BUDGET_BYTES, cap)

    /**
     * Created empty if missing; null where nothing runs. Holds no GPU memory until [reserve].
     * Throws if it can't be made (see [geometry]).
     */
    fun level(image: Image, pageScale: Float): Mipmap? {
        forgetReplacedDevice()
        val geometry = geometry(pageScale) ?: return null
        val base = image.mipmaps.firstOrNull() ?: return null
        val signature = signature(pageScale)

        var entry = levels[image]
        if (entry != null && (entry.signature != signature || entry.version != image.contentVersion ||
                    entry.k != geometry.k || entry.blockSize != geometry.blockSize)
        ) {
            drop(image)
            entry = null
        }

        if (entry == null) {
            val k = geometry.k
            val blockSize = geometry.blockSize
            val blockTexels = k * blockSize
            // Whole blocks per tile, so a block never straddles two.
            val most = min(WebGpuRenderer.maxTextureDimension2D, max(MAX_LEVEL_TILE, blockTexels))
            val tilesize = most / blockTexels * blockTexels
            check(tilesize >= blockTexels) {
                "ImageFilterChain [${names(pageScale).joinToString(", ")}]: a ${blockTexels}px block exceeds the texture limit"
            }
            entry = Level(
                image, signature, image.contentVersion, k, blockSize,
                Mipmap.writable(
                    k * base.width,
                    k * base.height,
                    k.toFloat(),
                    tilesize,
                    base.format
                ),
                (base.width + blockSize - 1) / blockSize,
                (base.height + blockSize - 1) / blockSize,
            )
            levels[image] = entry
        }
        return entry.mipmap
    }

    /** What a tile wants of [image]'s level: [rect], in image pixels. */
    class Wanted(val image: Image, val rect: ImageRect)

    /**
     * Allocates every level tile under [wanted], after [level], evicting tiles not read in [frame].
     * Throws if [wanted] alone exceeds the budget; false if it fits, but not beside this frame's.
     */
    fun reserve(wanted: List<Wanted>, pageScale: Float, frame: Long): Boolean {
        val need = LinkedHashSet<Slot>()
        val fresh = ArrayList<Slot>()
        var needBytes = 0L
        for ((image, rect) in wanted.map { it.image to it.rect }) {
            val entry = levels[image] ?: continue
            val per = entry.blocksPerTile
            val seen = HashSet<Int>()
            for (block in blocksIn(entry, rect)) {
                val column = block % entry.columns / per
                val row = block / entry.columns / per
                val i = row * entry.mipmap.tilesCols + column
                if (!seen.add(i)) continue
                val slot = entry.slots[i]
                if (slot != null) {
                    need.add(slot)
                    needBytes += slot.bytes
                } else {
                    val bytes = entry.mipmap.tileBytes(column, row)
                    fresh.add(Slot(entry, column, row, bytes))
                    needBytes += bytes
                }
            }
        }

        val budget = budget
        check(needBytes <= budget) {
            "ImageFilterChain [${names(pageScale).joinToString(", ")}]: a tile reads ${needBytes shr 20}MB " +
                    "of filtered level, past the ${budget shr 20}MB budget"
        }
        val freshBytes = fresh.sumOf { it.bytes }
        // Keep wanted levels even if emptied: the caller draws from them.
        val wantedLevels = wanted.mapNotNullTo(HashSet()) { levels[it.image] }
        for (slot in slots.toList()) {
            if (totalBytes + freshBytes <= budget) break
            // Gone already if an earlier release dropped its level.
            if (slot !in slots) continue
            if (slot.used != frame && slot !in need) release(slot, slot.level !in wantedLevels)
        }
        if (totalBytes + freshBytes > budget) return false

        val generation = generation
        for (slot in fresh) {
            slot.level.mipmap.allocate(slot.column, slot.row) {
                // Once per [clear]: the rest of a batch failing together adds nothing.
                if (generation == this.generation) outOfMemory()
            }
            slot.level.slots[slot.row * slot.level.mipmap.tilesCols + slot.column] = slot
            slots.add(slot)
            totalBytes += slot.bytes
            need.add(slot)
        }
        for (slot in need) {
            slot.used = frame
            slots.remove(slot)
            slots.add(slot)
        }
        return true
    }

    /** Halves the cap and drops what was made: some went into invalid textures. */
    private fun outOfMemory() {
        val cap = min(totalBytes, budget) / 2
        if (cap >= this.cap) return
        Log.w(TAG, "Out of memory, keeping ${cap shr 20}MB of levels")
        this.cap = cap
        changed()
    }

    /** Frees [slot]'s texture and unmarks its blocks; with [dropEmpty], drops its level if now empty. */
    private fun release(slot: Slot, dropEmpty: Boolean = true) {
        val level = slot.level
        val per = level.blocksPerTile
        for (r in slot.row * per until min(level.rows, (slot.row + 1) * per)) {
            for (c in slot.column * per until min(level.columns, (slot.column + 1) * per)) {
                level.filled[r * level.columns + c] = false
            }
        }
        level.mipmap.release(slot.column, slot.row)
        level.slots.remove(slot.row * level.mipmap.tilesCols + slot.column)
        slots.remove(slot)
        totalBytes -= slot.bytes
        // Nothing made is left in it.
        if (dropEmpty && level.slots.isEmpty() && levels[level.image] === level) drop(level.image)
    }

    private fun drop(image: Image) {
        val entry = levels.remove(image) ?: return
        for (slot in entry.slots.values.toList()) release(slot)
        entry.mipmap.cleanup()
    }

    /** For an image that is gone. */
    fun forget(image: Image) = drop(image)

    /** Drops levels of cleaned-up images. */
    fun prune() {
        for (image in levels.keys.toList()) {
            if (image.mipmaps.isEmpty()) drop(image)
        }
    }

    /** Blocks under [rect], as `row * columns + column`. */
    private fun blocksIn(entry: Level, rect: ImageRect): List<Int> {
        val size = entry.blockSize
        val c0 = max(0, floor(rect.left / size).toInt())
        val r0 = max(0, floor(rect.top / size).toInt())
        val c1 = min(entry.columns - 1, floor((rect.right - 1e-3f) / size).toInt())
        val r1 = min(entry.rows - 1, floor((rect.bottom - 1e-3f) / size).toInt())
        val blocks = ArrayList<Int>()
        for (r in r0..r1) {
            for (c in c0..c1) blocks.add(r * entry.columns + c)
        }
        return blocks
    }

    /** Unmade blocks under [rect]. Call after [level]. */
    fun missing(image: Image, rect: ImageRect): Int {
        val entry = levels[image] ?: return 0
        return blocksIn(entry, rect).count { !entry.filled[it] }
    }

    // Objects of a device that was lost and replaced are dropped, not closed: calling into them
    // crashed in native code.
    private var builtOn: GPUDevice? = null
    private var inputTexture: GPUTexture? = null
    private var inputView: GPUTextureView? = null
    private var inputSize = 0
    private var inputFormat = 0
    private var prepared: FilterOutput? = null

    private fun forgetReplacedDevice() {
        val device = device
        if (builtOn != null && builtOn !== device) {
            inputTexture = null
            inputView = null
            prepared = null
            levels.clear()
            slots.clear()
            totalBytes = 0
            cap = Long.MAX_VALUE
            placements.clear()
            writes.clear()
            probeTarget = null
            writers.built.forEach { forgetGroupLayout(it) }
            writers.clear()
        }
        builtOn = device
    }

    /**
     * Builds the block input, in [format], and binds each filter to the previous one's output. False if
     * nothing runs, or a filter can't be built, which sets [broken] unless [probing].
     */
    fun ready(
        pageScale: Float,
        format: Int,
        blockSize: Int = this.blockSize,
        probing: Boolean = false
    ): Boolean {
        val geometry = geometry(pageScale, blockSize) ?: return false
        prepared = null
        return try {
            forgetReplacedDevice()
            val size = geometry.side
            var texture = inputTexture
            if (texture == null || inputSize != size || inputFormat != format) {
                inputView?.close()
                texture?.destroyAndRelease()
                inputTexture = null
                inputView = null
                texture = device.createTexture(
                    GPUTextureDescriptor(
                        label = "image filter input",
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
                out = if (probing) f.prepareProbe(view, side, pageScale) else f.prepare(
                    view,
                    side,
                    pageScale
                )
                view = out.view
                side = f.outputSize(side, pageScale)
            }
            prepared = out
            out != null
        } catch (e: Exception) {
            if (probing) {
                Log.w(TAG, "A filter could not be prepared for a ${blockSize}px block", e)
            } else {
                Log.w(TAG, "A filter could not be prepared, drawing from the image", e)
                broken = true
            }
            false
        }
    }

    /** [fill]'s answer: blocks encoded, and what marks them made once submitted. */
    class Filled(val made: Int, val commit: () -> Unit)

    /**
     * Encodes up to [limit] of the unmade blocks under [rect], after [ready]. The first pass writes
     * [timestamps].
     * [first] is the first block's index among those in the caller's command buffer, for buffers.
     *
     * Marks nothing: call [Filled.commit] after submit, so an unsubmitted buffer leaves them to be remade.
     */
    fun fill(
        encoder: GPUCommandEncoder,
        image: Image,
        pageScale: Float,
        rect: ImageRect,
        timestamps: GPUPassTimestampWrites?,
        first: Int,
        limit: Int = Int.MAX_VALUE,
    ): Filled {
        val entry = levels[image]
        val geometry = geometry(pageScale)
        val output = prepared
        val input = inputTexture
        val inputView = inputView
        if (entry == null || geometry == null || output == null || input == null || inputView == null ||
            entry.blockSize != geometry.blockSize || inputSize != geometry.side
        ) {
            return Filled(0) {}
        }
        val k = geometry.k
        val halo = geometry.halo
        val blockTexels = k * geometry.blockSize
        val tilesize = entry.mipmap.tilesize
        // Skips blocks whose tile isn't [reserve]d.
        val todo = blocksIn(entry, rect).filter { i ->
            if (entry.filled[i]) return@filter false
            val px = i % entry.columns * blockTexels
            val py = i / entry.columns * blockTexels
            entry.mipmap.isAllocated(px / tilesize, py / tilesize)
        }.take(limit)
        val running = running(pageScale)

        Hdr.cuttingTiles = true
        try {
            todo.forEachIndexed { n, index ->
                val column = index % entry.columns
                val row = index / entry.columns
                drawBlock(
                    encoder, image, column, row, geometry, input, inputView, first + n,
                    if (n == 0) timestamps else null,
                )
                for (f in running) f.encode(encoder)

                val px = column * blockTexels
                val py = row * blockTexels
                val tileColumn = px / tilesize
                val tileRow = py / tilesize
                val (target, targetView) = entry.mipmap.target(tileColumn, tileRow)
                val x0 = px - tileColumn * tilesize
                val y0 = py - tileRow * tilesize
                val width = min(blockTexels, target.width - x0)
                val height = min(blockTexels, target.height - y0)
                val write = encoder.beginRenderPass(
                    GPURenderPassDescriptor(
                        colorAttachments = arrayOf(
                            GPURenderPassColorAttachment(
                                view = targetView, loadOp = LoadOp.Load, storeOp = StoreOp.Store,
                                clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                            )
                        )
                    )
                )
                try {
                    write.setViewport(
                        x0.toFloat(),
                        y0.toFloat(),
                        width.toFloat(),
                        height.toFloat(),
                        0f,
                        1f
                    )
                    write.setScissorRect(x0, y0, width, height)
                    encodeWrite(write, target.format, output.view, first + n, x0, y0, k * halo)
                } finally {
                    write.endAndRelease()
                }
            }
        } finally {
            Hdr.cuttingTiles = false
        }

        // A filter failing in encode leaves these unmade.
        val made = running.all { it.supported }
        return Filled(todo.size) {
            if (made) todo.forEach { entry.filled[it] = true }
        }
    }

    /** Block ([column], [row]) plus halo at 1:1, edges extended so the network sees no blank. */
    private fun drawBlock(
        encoder: GPUCommandEncoder,
        image: Image,
        column: Int,
        row: Int,
        geometry: BlockGeometry,
        input: GPUTexture,
        inputView: GPUTextureView,
        i: Int,
        timestamps: GPUPassTimestampWrites?,
    ) {
        val base = image.mipmaps[0]
        val halo = geometry.halo
        val side = geometry.side
        val draw = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = inputView, loadOp = LoadOp.Clear, storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                ),
                timestampWrites = timestamps,
            )
        )
        try {
            val originX = column * geometry.blockSize - halo
            val originY = row * geometry.blockSize - halo
            // Centre putting the left edge at -originX: see solveImagePlacement.
            val x =
                (-originX + base.width / 2f - side / 2f) / side - image.x / side - WebGpuRenderer.offsetX
            val y =
                (-originY + base.height / 2f - side / 2f) / side - image.y / side - WebGpuRenderer.offsetY
            val h = halo.toFloat()
            RenderPage.render(
                draw,
                image,
                input,
                x,
                y,
                1f,
                firstStep(),
                Extend(h, h, h, h),
                true,
                placement(i)
            )
        } finally {
            draw.endAndRelease()
        }
    }

    /**
     * Runs the chain on [image]'s first [blockSize] block, keeping nothing, to time a size not in use.
     * The first pass writes [timestamps]. False if the chain can't run at that size.
     */
    fun probe(
        encoder: GPUCommandEncoder,
        image: Image,
        pageScale: Float,
        blockSize: Int,
        timestamps: GPUPassTimestampWrites,
    ): Boolean {
        val base = image.mipmaps.firstOrNull() ?: return false
        val geometry = try {
            geometry(pageScale, blockSize)
        } catch (e: IllegalStateException) {
            null
        } ?: return false
        if (!ready(pageScale, base.format, blockSize, probing = true)) return false
        val output = prepared ?: return false
        Hdr.cuttingTiles = true
        try {
            drawBlock(encoder, image, 0, 0, geometry, inputTexture!!, inputView!!, 0, timestamps)
        } finally {
            Hdr.cuttingTiles = false
        }
        for (f in running(pageScale)) f.encode(encoder)
        // A real block's write too, or a probed size reads as cheaper than the one in use.
        val texels = geometry.k * blockSize
        val target = probeTarget(texels, base.format)
        val write = encoder.beginRenderPass(
            GPURenderPassDescriptor(
                colorAttachments = arrayOf(
                    GPURenderPassColorAttachment(
                        view = target.view, loadOp = LoadOp.Clear, storeOp = StoreOp.Store,
                        clearValue = GPUColor(0.0, 0.0, 0.0, 0.0)
                    )
                )
            )
        )
        try {
            encodeWrite(write, base.format, output.view, 0, 0, 0, geometry.k * geometry.halo)
        } finally {
            write.endAndRelease()
        }
        // Prepared for this size, not the one in use: [fill] waits for the next [ready].
        prepared = null
        return true
    }

    private var probeTarget: FilterOutput? = null

    /** What [probe] writes to, kept while the size and format are. */
    private fun probeTarget(side: Int, format: Int): FilterOutput {
        probeTarget?.let {
            if (it.texture.width == side && it.texture.format == format) return it
            it.view.close()
            it.texture.destroyAndRelease()
            probeTarget = null
        }
        val texture = device.createTexture(
            GPUTextureDescriptor(
                label = "image filter probe", size = GPUExtent3D(side, side), format = format,
                usage = TextureUsage.RenderAttachment,
            )
        )
        return FilterOutput(texture, texture.createView()).also { probeTarget = it }
    }

    private var firstStepPair: RenderPage.Filtered? = null

    /** At 1:1 on the pixel grid, Catmull-Rom is a copy. */
    private fun firstStep(): RenderPage.Filtered = firstStepPair
        ?: RenderPage.filtered(Sampler.catmullRom().code, Sampler.box().code)
            .also { firstStepPair = it }

    /**
     * For block [i] of a command buffer: writes land before it runs, so each draw needs its own,
     * not the image's. Reusable by the next buffer: its writes queue after this one's commands.
     */
    private fun placement(i: Int): GPUBuffer {
        while (placements.size <= i) {
            placements.add(
                device.createBuffer(
                    GPUBufferDescriptor(
                        label = "image filter block placement", size = BUFFER_SIZE,
                        usage = BufferUsage.CopyDst or BufferUsage.Uniform,
                    )
                )
            )
        }
        return placements[i]
    }

    private val placements = ArrayList<GPUBuffer>()
    private val writes = ArrayList<GPUBuffer>()
    // Exactly the uniform's size: writeBuffer uploads a ByteBuffer's whole capacity.
    private val writeBytes = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())

    private fun encodeWrite(
        pass: GPURenderPassEncoder,
        format: Int,
        source: GPUTextureView,
        i: Int,
        x0: Int,
        y0: Int,
        inset: Int,
    ) {
        val writer = writers[format]
        while (writes.size <= i) {
            writes.add(
                device.createBuffer(
                    GPUBufferDescriptor(
                        label = "image filter write",
                        size = 16,
                        usage = BufferUsage.CopyDst or BufferUsage.Uniform
                    )
                )
            )
        }
        val buffer = writes[i]
        writeBytes.clear()
        writeBytes.putInt(x0).putInt(y0).putInt(inset).putInt(inset)
        writeBytes.flip()
        device.queue.writeBuffer(buffer, 0, writeBytes)
        pass.setPipeline(writer)
        pass.setTransientBindGroup(
            0, device.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = writer.groupLayout(), label = "image filter write", entries = arrayOf(
                        GPUBindGroupEntry(0, textureView = source),
                        GPUBindGroupEntry(1, buffer = buffer),
                    )
                )
            )
        )
        pass.draw(3)
    }

    private var generation = 0

    /** Drops every level, keeping what [ready] built. */
    fun clear() {
        generation++
        for (image in levels.keys.toList()) drop(image)
    }

    /** Frees everything; [ready] rebuilds. */
    fun cleanup() {
        clear()
        blockSize = BLOCK_SIZE
        cap = Long.MAX_VALUE
        filters.forEach { it.cleanup() }
        inputView?.close()
        inputTexture?.destroyAndRelease()
        inputTexture = null
        inputView = null
        prepared = null
        placements.forEach { it.destroyAndRelease() }
        placements.clear()
        writes.forEach { it.destroyAndRelease() }
        writes.clear()
        probeTarget?.let {
            it.view.close()
            it.texture.destroyAndRelease()
        }
        probeTarget = null
    }

    private companion object {
        const val TAG = "ImageFilterChain"

        // Shared by every chain, so never closed.
        val writers = FormatKeyed { format ->
            Fullscreen.buildPipeline(
                write(if (format == TextureFormat.RGBA8Unorm) "1.0" else "65504.0"),
                format,
                "image filter write"
            )
        }
    }
}
