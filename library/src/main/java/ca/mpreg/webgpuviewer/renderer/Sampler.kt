package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.ErrorFilter
import kotlinx.coroutines.withContext

/**
 * The WGSL [TileRenderer] resolves a high-quality tile with in one step, through [RenderPage.render].
 *
 * [code] defines `resolve(src_start, scale, bounds) -> vec4<f32>`: the top-left of one destination
 * pixel's footprint in source texels, its size in texels (under 1 per axis enlarges), and the first
 * (xy) and last (zw) texel it may read, a `vec4<i32>`, to colour as texels are stored: sRGB-encoded,
 * straight alpha, extended-sRGB in an HDR tile. It reads only through the host's
 * `load(texel) -> vec4<f32>`, which gives them so, and only within bounds: past them, it clamps.
 * Which space it filters in is its own choice. Filtering colour without weighting by alpha bleeds
 * transparent texels' colour.
 *
 * Any sampler handles any ratio, so either of [TileRenderer]'s two roles may take it. Networks that
 * change resolution are [ImageFilter]s instead.
 *
 * [name] is for display.
 */
class Sampler(val code: String, val name: String) {
    /**
     * Compiles [code] against the host, so a shader the device rejects fails here, not on first
     * use. Does nothing without WebGPU.
     *
     * @throws IllegalArgumentException with the device's message.
     */
    suspend fun validate() {
        if (!WebGpuRenderer.isAvailable) return
        val error = withContext(WebGpuRenderer.dispatcher) {
            val device = WebGpuRenderer.device
            device.pushErrorScope(ErrorFilter.Validation)
            val built = try {
                RenderPage.buildResolve(code)
                null
            } catch (e: Exception) {
                e.message ?: e.toString()
            }
            // Popped before anything suspends: a frame's errors would land in the scope.
            val pop = ErrorScopes.popMessage(device)
            built ?: ErrorScopes.pumpingEvents { pop.await() }
        }
        require(error == null) { error.orEmpty() }
    }

    companion object {
        const val CATMULL_ROM_FILE = "catmull_rom.wgsl"
        const val BOX_FILE = "box_downscale.wgsl"

        private val CATMULL_ROM by lazy { readWgslResource(CATMULL_ROM_FILE) }
        private val BOX by lazy { readWgslResource(BOX_FILE) }

        /** Catmull-Rom, kernel stretched where a pixel covers more than a texel. */
        fun catmullRom() = Sampler(CATMULL_ROM, CATMULL_ROM_FILE)

        /** Area-weighted average of the source pixels each destination pixel covers. */
        fun box() = Sampler(BOX, BOX_FILE)
    }
}
