package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUDevice

// Compute pipelines shared across the process by WGSL source.
internal object ComputePipelines {
    private const val MAX_IDLE = 32

    private val cache = RefCache<GPUComputePipeline>(
        MAX_IDLE,
        // A lost device's objects crash when called into.
        close = { if (WebGpuRenderer.isAvailable) it.closeWithLayout() else forgetGroupLayout(it) },
        forget = { forgetGroupLayout(it) },
    )

    private var builtOn: GPUDevice? = null

    // Pair with [release].
    @Synchronized
    fun acquire(code: String, build: () -> GPUComputePipeline): RefCache.Entry<GPUComputePipeline> {
        val device = WebGpuRenderer.device
        if (builtOn != null && builtOn !== device) cache.clear()
        builtOn = device
        return cache.acquire(code, build)
    }

    @Synchronized
    fun release(entry: RefCache.Entry<GPUComputePipeline>) = cache.release(entry)

    @Synchronized
    fun evict(entry: RefCache.Entry<GPUComputePipeline>) = cache.evict(entry)
}

/**
 * Reference-counted values by key. Unheld ones stay among the [maxIdle] most recently released,
 * closed past that. Not thread-safe.
 */
internal class RefCache<V : Any>(
    private val maxIdle: Int,
    private val close: (V) -> Unit,
    /** For [clear]ed values: dropped without [close]. */
    private val forget: (V) -> Unit,
) {
    class Entry<V : Any> internal constructor(internal val key: String, val value: V) {
        internal var users = 0
        internal var cleared = false
    }

    private val entries = HashMap<String, Entry<V>>()

    // Unheld entries, least recently released first.
    private val idle = LinkedHashMap<String, Entry<V>>()

    fun acquire(key: String, build: () -> V): Entry<V> {
        val entry = entries.getOrPut(key) { Entry(key, build()) }
        if (entry.users++ == 0) idle.remove(key)
        return entry
    }

    fun release(entry: Entry<V>) {
        check(entry.users > 0) { "released more often than acquired" }
        if (--entry.users > 0) return
        when {
            entry.cleared -> forget(entry.value)
            entries[entry.key] !== entry -> close(entry.value)
            else -> {
                idle[entry.key] = entry
                while (idle.size > maxIdle) {
                    val oldest = idle.values.first()
                    idle.remove(oldest.key)
                    entries.remove(oldest.key)
                    close(oldest.value)
                }
            }
        }
    }

    /** Hand [entry] out no more; closed now if unheld, else at its last [release]. */
    fun evict(entry: Entry<V>) {
        if (entries[entry.key] !== entry) return
        entries.remove(entry.key)
        if (idle.remove(entry.key) != null) close(entry.value)
    }

    /** Drop everything without closing it, held entries at their last [release]. */
    fun clear() {
        for (entry in entries.values) {
            if (entry.users == 0) forget(entry.value) else entry.cleared = true
        }
        entries.clear()
        idle.clear()
    }
}
