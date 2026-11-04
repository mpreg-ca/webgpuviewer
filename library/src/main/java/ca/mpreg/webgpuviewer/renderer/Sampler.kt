package ca.mpreg.webgpuviewer.renderer

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
 */
class Sampler(val code: String) {
    companion object {
        private val CATMULL_ROM by lazy { readWgslResource("catmull_rom.wgsl") }
        private val BOX by lazy { readWgslResource("box_downscale.wgsl") }

        /** Catmull-Rom, kernel stretched where a pixel covers more than a texel. */
        fun catmullRom() = Sampler(CATMULL_ROM)

        /** Area-weighted average of the source pixels each destination pixel covers. */
        fun box() = Sampler(BOX)
    }
}
