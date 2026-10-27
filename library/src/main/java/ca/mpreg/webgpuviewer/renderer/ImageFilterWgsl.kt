package ca.mpreg.webgpuviewer.renderer

import android.util.Log
import androidx.webgpu.ErrorFilter
import androidx.webgpu.ErrorType
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUComputePassDescriptor
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUPassTimestampWrites
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.GPUSampler
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.WebGpuRuntimeException
import ca.mpreg.webgpuviewer.renderer.ImageFilterWgsl.Companion.TEST_SIDE
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.time.Duration.Companion.milliseconds

/**
 * The only [ImageFilter]: passes, textures, bindings, sizes and the scales it runs at all come from
 * WGSL source with mpv-user-shader-style pass headers, run per image block.
 *
 * Nothing here knows the network. `HOOKED` is the block, a pass without `//!SAVE` replaces it, and
 * textures between passes are sized from the block and reused once their last reader has run.
 *
 * A block runs the passes whose `//!WHEN` accepts the page scale, or none. Bindings hold while the
 * block size and those passes do, so a block costs one `dispatchWorkgroups` per pass in one
 * compute pass.
 *
 * Holds GPU objects for one chain's block size: one viewer per instance. If the device can't
 * build it, or runs out of memory doing so, [supported] goes false for good.
 *
 * @throws IllegalArgumentException if [source] is not in that format.
 */
class ImageFilterWgsl(source: String, private val label: String = "Wgsl") : ImageFilter() {

    private val network = WgslNetwork.parse(source)

    init {
        require(network.stages.isNotEmpty()) { "$label: no pass" }
    }

    /**
     * Runs the network once on a blank [TEST_SIDE]-pixel tile, so a shader the device rejects
     * fails here, with the device's message, not as [supported] going false on first use. Call
     * before giving it to a viewer. Does nothing without WebGPU.
     *
     * @throws IllegalArgumentException with the device's message.
     */
    suspend fun validate() {
        if (!WebGpuRenderer.isAvailable) return
        val scale = TEST_SCALES.firstOrNull { applies(it) } ?: return
        val error = withContext(WebGpuRenderer.dispatcher) {
            pumpingEvents {
                forgetReplacedDevice()
                try {
                    testRun(scale).also { error ->
                        if (error != null) pipelines.values.forEach { ComputePipelines.evict(it) }
                    }
                } finally {
                    cleanup()
                }
            }
        }
        require(error == null) { error.orEmpty() }
    }

    /** The first error building and running [pageScale]'s passes, or null. */
    private suspend fun testRun(pageScale: Float): String? {
        val plan = try {
            network.plan(info(pageScale).active, TEST_SIDE, pageScale)
        } catch (e: IllegalArgumentException) {
            return e.message
        }
        val device = device
        val owned = ArrayList<() -> Unit>()
        val filters =
            intArrayOf(ErrorFilter.Internal, ErrorFilter.OutOfMemory, ErrorFilter.Validation)
        filters.forEach { device.pushErrorScope(it) }
        var error: String? = null
        try {
            val input = device.createTexture(
                GPUTextureDescriptor(
                    label = "$label test input", size = GPUExtent3D(TEST_SIDE, TEST_SIDE),
                    format = TextureFormat.RGBA8Unorm, usage = TextureUsage.TextureBinding,
                )
            )
            owned.add { input.destroyAndRelease() }
            val view = input.createView().also { view -> owned.add { view.close() } }
            val textures = Textures(plan, view, owned)
            val encoder = device.createCommandEncoder()
            dispatch(encoder, textures, null)
            device.queue.submitAndRelease(encoder)
        } catch (e: IllegalArgumentException) {
            error = e.message
        } finally {
            // Popped before anything suspends: frames run meanwhile, and their errors would land in
            // these scopes. Innermost first.
            val pops = List(filters.size) { popScope(device) }
            try {
                device.queue.onSubmittedWorkDone()
                for (pop in pops) {
                    val message = pop.await()
                    if (error == null) error = message
                }
            } finally {
                release(owned)
            }
        }
        return error
    }

    /** The innermost scope's error message, or null. Pops at once; the answer needs events processed. */
    private fun popScope(device: GPUDevice): Deferred<String?> {
        val answer = CompletableDeferred<String?>()
        device.popErrorScope(Executor(Runnable::run), object : GPURequestCallback<Int> {
            override fun onResult(result: Int) {
                answer.complete(if (result == ErrorType.NoError) null else ErrorType.toString(result))
            }

            override fun onError(exception: Exception) {
                // Else a lost device, which fails the work itself.
                answer.complete((exception as? WebGpuRuntimeException)?.message)
            }
        })
        return answer
    }

    /** Async WebGPU calls resolve only while events are processed. */
    private suspend fun <T> pumpingEvents(block: suspend () -> T): T = coroutineScope {
        val pump = launch {
            while (isActive) {
                WebGpuRenderer.instance.processEvents()
                delay(1.milliseconds)
            }
        }
        try {
            block()
        } finally {
            pump.cancel()
        }
    }

    override val supported: Boolean get() = !failed

    @Volatile
    private var failed = false

    private class ScaleInfo(
        val active: List<WgslNetwork.Stage>, val usable: Boolean, val scale: Float, val halo: Int
    )

    /**
     * The passes whose `//!WHEN` accept [pageScale], and whether they can run: reading a skipped
     * pass's output, or a non-square result, leaves the filter out at that scale, not failed for
     * good. Cached: asked per tile, and per frame a plain tile awaits an upgrade.
     */
    private fun info(pageScale: Float): ScaleInfo {
        infos[pageScale]?.let { return it }
        val active = network.active(pageScale)
        var found = ScaleInfo(active, false, 1f, 0)
        if (active.isNotEmpty()) {
            try {
                val result = network.plan(active, REFERENCE_SIDE, pageScale).result
                require(result.width == result.height) {
                    "writes ${result.width}x${result.height}, which is not square"
                }
                found = ScaleInfo(
                    active, true, result.width.toFloat() / REFERENCE_SIDE, network.halo(active)
                )
            } catch (e: IllegalArgumentException) {
                Log.w(TAG, "$label left out at scale $pageScale", e)
            }
        }
        // Scale varies continuously in a pinch; unbounded, this grows all session.
        if (infos.size >= MAX_INFOS) infos.clear()
        infos[pageScale] = found
        return found
    }

    private val infos = HashMap<Float, ScaleInfo>()

    private class KeptPlan(val size: Int, val scale: Float, val plan: WgslNetwork.Plan)

    /** [info]'s passes sized for a [size]-square input; caches the last. */
    private fun planFor(size: Int, pageScale: Float): WgslNetwork.Plan {
        lastPlan?.let { if (it.size == size && it.scale == pageScale) return it.plan }
        val plan = network.plan(info(pageScale).active, size, pageScale)
        lastPlan = KeptPlan(size, pageScale, plan)
        return plan
    }

    private var lastPlan: KeptPlan? = null

    override fun applies(pageScale: Float) = info(pageScale).usable

    override fun scale(pageScale: Float) = info(pageScale).scale

    override fun outputSize(size: Int, pageScale: Float) =
        if (info(pageScale).usable) planFor(size, pageScale).result.width else size

    override fun halo(pageScale: Float) = info(pageScale).halo

    override fun fits(size: Int, pageScale: Float): Boolean {
        if (!info(pageScale).usable) return false
        return try {
            val plan = planFor(size, pageScale)
            plan.result.width == plan.result.height && memoryError(plan) == null
        } catch (e: IllegalArgumentException) {
            // Size expressions fail for this size and scale.
            false
        }
    }

    override fun signature(pageScale: Float): String {
        // By index: `//!DESC` need not be unique.
        val stages = network.stages
        return "$label[${
            info(pageScale).active.joinToString(",") {
                stages.indexOf(it).toString()
            }
        }]"
    }

    override fun prepare(input: GPUTextureView, size: Int, pageScale: Float) =
        prepare(input, size, pageScale, false)

    override fun prepareProbe(input: GPUTextureView, size: Int, pageScale: Float) =
        prepare(input, size, pageScale, true)

    private fun prepare(
        input: GPUTextureView,
        size: Int,
        pageScale: Float,
        probing: Boolean
    ): FilterOutput {
        check(!failed) { "$label unavailable" }
        try {
            val plan = planFor(size, pageScale)
            forgetReplacedDevice()
            var t = textures
            if (t == null || t.input !== input || t.signature != plan.signature) {
                // Cleared first, so [fail] can't destroy these twice if the build throws.
                textures = null
                t?.destroy()
                // Here, not in [encode], so a throw lets the caller fall back. GPU errors arrive
                // later via [fail], which drops what was made with it.
                t = create(plan, input, probing)
                textures = t
            }
            return t.output
        } catch (e: Throwable) {
            if (probing) Log.w(TAG, "$label can't run at ${size}px", e) else fail(e)
            throw e
        }
    }

    override val name: String get() = label

    override fun encode(encoder: GPUCommandEncoder, timestampWrites: GPUPassTimestampWrites?) {
        val t = textures ?: return
        if (failed) return
        try {
            dispatch(encoder, t, timestampWrites)
        } catch (e: Exception) {
            fail(e)
        }
    }

    private fun dispatch(
        encoder: GPUCommandEncoder,
        t: Textures,
        timestampWrites: GPUPassTimestampWrites?
    ) {
        // One pass: WebGPU orders dispatches and inserts barriers within it, and a begin/end
        // per stage per tile is real work on a tiler.
        val compute = encoder.beginComputePass(
            GPUComputePassDescriptor(label = label, timestampWrites = timestampWrites)
        )
        try {
            for (p in t.dispatches) {
                compute.setPipeline(p.pipeline)
                compute.setBindGroup(0, p.bindGroup)
                compute.dispatchWorkgroups(p.groupsX, p.groupsY)
            }
        } finally {
            compute.endAndRelease()
        }
    }

    /** Frees everything; [prepare] rebuilds unless [failed]. */
    override fun cleanup() {
        textures?.destroy()
        textures = null
        pipelines.values.forEach { ComputePipelines.release(it) }
        pipelines.clear()
        gpuSampler?.close()
        gpuSampler = null
        lastPlan = null
    }

    /** For good: the chain skips this via [supported]. */
    private fun fail(e: Throwable) {
        if (failed) return
        failed = true
        Log.w(TAG, "$label unavailable", e)
        // One may be what failed; don't hand it to another reader.
        pipelines.values.forEach { ComputePipelines.evict(it) }
        try {
            cleanup()
        } catch (cleanupError: Exception) {
            Log.w(TAG, "$label cleanup after failure", cleanupError)
        }
        // Deferred: this can be mid-block, and the holder drops everything made with it.
        WebGpuRenderer.dispatcher.dispatch(EmptyCoroutineContext) { onUpdate?.invoke() }
    }

    // Objects of a device that was lost and replaced are dropped, not closed: calling into them
    // crashed in native code.
    private var builtOn: GPUDevice? = null

    private fun forgetReplacedDevice() {
        val current = device
        if (builtOn != null && builtOn !== current) {
            textures = null
            // [ComputePipelines] forgets them itself.
            pipelines.clear()
            gpuSampler = null
        }
        builtOn = current
    }

    // Held from [ComputePipelines] until [cleanup].
    private val pipelines = HashMap<WgslNetwork.Stage, RefCache.Entry<GPUComputePipeline>>()

    private fun pipelineFor(stage: WgslNetwork.Stage): GPUComputePipeline =
        pipelines.getOrPut(stage) { ComputePipelines.acquire(stage.code) { build(stage) } }.value

    private fun build(stage: WgslNetwork.Stage): GPUComputePipeline {
        val module = device.createShaderModule(
            GPUShaderModuleDescriptor(
                label = stage.name,
                shaderSourceWGSL = GPUShaderSourceWGSL(stage.code)
            )
        )
        // The pipeline keeps its own reference to the module.
        try {
            return device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    label = "$label ${stage.name}",
                    compute = GPUComputeState(module, entryPoint = "main")
                )
            )
        } finally {
            module.close()
        }
    }

    private var gpuSampler: GPUSampler? = null

    private fun linearSampler(): GPUSampler = gpuSampler ?: device.createSampler(
        GPUSamplerDescriptor(
            label = label,
            magFilter = FilterMode.Linear,
            minFilter = FilterMode.Linear
        )
    ).also { gpuSampler = it }

    private var textures: Textures? = null

    /** One dispatch and its fixed bindings. */
    private class Dispatch(
        val pipeline: GPUComputePipeline,
        val bindGroup: GPUBindGroup,
        val groupsX: Int,
        val groupsY: Int
    )

    /**
     * Builds [Textures] so a partial failure frees what was made: the constructor registers each
     * GPU object in `owned`, as the half-built instance is unreachable once it throws.
     */
    private fun create(plan: WgslNetwork.Plan, input: GPUTextureView, probing: Boolean): Textures {
        val device = device
        val owned = ArrayList<() -> Unit>()
        var built: Textures? = null
        // A bad shader, binding or OOM only shows later as an invalid object, so scope it; the
        // scopes close once everything is built and answer later.
        device.pushErrorScope(ErrorFilter.Validation)
        device.pushErrorScope(ErrorFilter.OutOfMemory)
        try {
            return Textures(plan, input, owned).also { built = it }
        } catch (e: Throwable) {
            release(owned)
            throw e
        } finally {
            repeat(2) {
                ErrorScopes.pop(device) { type ->
                    val error = IllegalStateException("${ErrorType.toString(type)} building $label")
                    if (!probing) {
                        fail(error)
                        return@pop
                    }
                    // A size only being timed: drop what it made, if still held, and carry on.
                    Log.w(TAG, "$label probe", error)
                    val made = built
                    if (made != null && textures === made) {
                        textures = null
                        made.destroy()
                    }
                }
            }
        }
    }

    /** Why [plan]'s textures can't be built, or null: fails a bad file or huge block before the GPU does. */
    private fun memoryError(plan: WgslNetwork.Plan): String? {
        val limit = WebGpuRenderer.maxTextureDimension2D
        var bytes = 0L
        if (plan.input.width !in 1..limit) {
            return "$label: a ${plan.input.width}-texel block is past the device's limit"
        }
        for (slot in plan.slots) {
            if (slot.width > limit || slot.height > limit) {
                return "$label: a ${slot.width}x${slot.height} texture is past the device's limit"
            }
            bytes += slot.width.toLong() * slot.height * WgslNetwork.bytesPerTexel(slot.format)
        }
        if (bytes > MAX_TEXTURE_BYTES) {
            return "$label: ${plan.input.width}-texel blocks need ${bytes shr 20}MB of textures"
        }
        return null
    }

    /** Newest first: bind groups before the views they hold, views before their textures. */
    private fun release(owned: MutableList<() -> Unit>) {
        for (i in owned.indices.reversed()) {
            try {
                owned[i]()
            } catch (e: Exception) {
                Log.w(TAG, "$label release", e)
            }
        }
        owned.clear()
    }

    /**
     * One plan's textures and bind groups. Rebuilt only when the (fixed) block size or the running
     * passes change.
     *
     * Each is a Dawn handle of its own, and only close() drops it - nothing releases one when it
     * is garbage collected.
     */
    private inner class Textures(
        plan: WgslNetwork.Plan,
        val input: GPUTextureView,
        private val owned: MutableList<() -> Unit>
    ) {
        val signature = plan.signature
        val dispatches: List<Dispatch>

        /** What the last pass to replace `HOOKED` writes. */
        val output: FilterOutput

        init {
            memoryError(plan)?.let { throw IllegalArgumentException(it) }

            // CopySrc is only for the one a cache copies from; the rest share a layout.
            val slotOutputs = plan.slots.map { slot ->
                val texture = device.createTexture(
                    GPUTextureDescriptor(
                        label = "$label feature", size = GPUExtent3D(slot.width, slot.height),
                        format = slot.format,
                        usage = TextureUsage.StorageBinding or TextureUsage.TextureBinding or TextureUsage.CopySrc,
                    )
                )
                owned.add { texture.destroyAndRelease() }
                val view = texture.createView().also { view -> owned.add { view.close() } }
                FilterOutput(texture, view)
            }

            fun view(ref: WgslNetwork.Ref): GPUTextureView =
                if (ref === plan.input) input else slotOutputs[ref.slot].view

            val linear = linearSampler()
            dispatches = plan.steps.map { step ->
                val pipeline = pipelineFor(step.stage)
                var sampled = 0
                val entries = Array(step.stage.bindings.size) { j ->
                    val binding = step.stage.bindings[j]
                    when (binding.kind) {
                        WgslNetwork.Kind.Sampled ->
                            GPUBindGroupEntry(
                                binding.index,
                                textureView = view(step.reads[sampled++])
                            )

                        WgslNetwork.Kind.Storage ->
                            GPUBindGroupEntry(binding.index, textureView = view(step.writes))

                        WgslNetwork.Kind.Sampler -> GPUBindGroupEntry(
                            binding.index,
                            sampler = linear
                        )
                    }
                }
                val group = device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.groupLayout(),
                        label = label,
                        entries = entries
                    )
                )
                owned.add { group.close() }
                Dispatch(
                    pipeline, group,
                    (step.width + step.stage.blockX - 1) / step.stage.blockX,
                    (step.height + step.stage.blockY - 1) / step.stage.blockY,
                )
            }
            output = slotOutputs[plan.result.slot]
        }

        fun destroy() = release(owned)
    }

    companion object {
        private const val TAG = "ImageFilterWgsl"

        /** Far above any block's need; bounds a bad file. */
        private const val MAX_TEXTURE_BYTES = 256L shl 20

        /** Input side [scale] is measured at. */
        private const val REFERENCE_SIDE = 1024

        /** Scales [info] caches. */
        private const val MAX_INFOS = 8

        /** [validate] runs at the first of these the network applies at. */
        private val TEST_SCALES = floatArrayOf(1f, 2f, 0.5f)

        private const val TEST_SIDE = 16

        /** The filter in the library's WGSL resource [file]. */
        fun load(file: String, label: String = file) =
            ImageFilterWgsl(readWgslResource(file), label)
    }
}
