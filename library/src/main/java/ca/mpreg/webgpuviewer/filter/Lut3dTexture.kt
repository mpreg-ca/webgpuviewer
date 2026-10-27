package ca.mpreg.webgpuviewer.filter

import androidx.webgpu.AddressMode
import androidx.webgpu.FilterMode
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUSampler
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureBindingViewDimension
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.TextureDimension
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.TextureViewDimension
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.renderer.destroyAndRelease
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicReference

/**
 * A [Lut3d] as a 3D texture, for [ca.mpreg.webgpuviewer.renderer.TileRenderer]'s image LUT. Bind
 * [view] and [sampler] where [wgsl] declares them. [lut] may be set from any thread; [prepare]
 * uploads on the render thread.
 */
internal class Lut3dTexture {
    private val device get() = WebGpuRenderer.device

    @Volatile
    var lut: Lut3d? = null
        set(value) {
            field = value
            pendingRef.set(Pending(value))
        }

    // Wrapped so null means "nothing pending", apart from a pending null that turns it off.
    private class Pending(val lut: Lut3d?)

    private val pendingRef = AtomicReference<Pending?>(null)

    // What draws apply, which may trail [lut]. Render thread only.
    var applied: Lut3d? = null
        private set

    private var texture: GPUTexture? = null

    // Non-null while [applied] is.
    var view: GPUTextureView? = null
        private set

    private var lutSize = 0

    // Trilinear between entries, clamped so the outermost half-texel doesn't wrap around.
    val sampler: GPUSampler by lazy {
        device.createSampler(
            GPUSamplerDescriptor(
                magFilter = FilterMode.Linear,
                minFilter = FilterMode.Linear,
                addressModeU = AddressMode.ClampToEdge,
                addressModeV = AddressMode.ClampToEdge,
                addressModeW = AddressMode.ClampToEdge,
            )
        )
    }

    // True when [view] was replaced: rebuild bind groups holding the old one.
    fun prepare(): Boolean {
        val pending = pendingRef.getAndSet(null) ?: return false
        applied = pending.lut
        return upload(pending.lut ?: return release())
    }

    fun cleanup() {
        release()
        pendingRef.compareAndSet(null, Pending(applied))
        applied = null
    }

    // Not kept while off: a 128-point table is 16MB. True when there was a texture.
    private fun release(): Boolean {
        val had = texture != null
        view?.close()
        texture?.destroyAndRelease()
        texture = null
        view = null
        // Not just the texture: upload() skips creating one when the size already matches.
        lutSize = 0
        return had
    }

    private fun upload(lut: Lut3d): Boolean {
        val replaced = lutSize != lut.size
        if (replaced) {
            view?.close()
            texture?.destroyAndRelease()
            texture = device.createTexture(
                GPUTextureDescriptor(
                    label = LABEL,
                    size = GPUExtent3D(lut.size, lut.size, lut.size),
                    dimension = TextureDimension._3D,
                    format = TextureFormat.RGBA16Float,
                    usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
                    // Compat mode pins a texture to one view dimension; say which up front.
                    textureBindingViewDimension = GPUTextureBindingViewDimension(
                        TextureViewDimension._3D
                    ),
                )
            )
            view = texture!!.createView()
            lutSize = lut.size
        }

        val bytes = ByteBuffer.allocateDirect(lut.size * lut.size * lut.size * 8)
            .order(ByteOrder.nativeOrder())
        var i = 0
        while (i < lut.data.size) {
            bytes.putShort(half(lut.data[i]))
            bytes.putShort(half(lut.data[i + 1]))
            bytes.putShort(half(lut.data[i + 2]))
            bytes.putShort(ONE_HALF)
            i += 3
        }
        bytes.flip()

        device.queue.writeTexture(
            GPUTexelCopyTextureInfo(texture!!),
            bytes,
            GPUExtent3D(lut.size, lut.size, lut.size),
            GPUTexelCopyBufferLayout(
                bytesPerRow = lut.size * 8, rowsPerImage = lut.size
            )
        )
        return replaced
    }

    companion object {
        private const val LABEL = "image LUT"

        /** 1.0 as an IEEE half. */
        private const val ONE_HALF: Short = 0x3c00

        /** Float to IEEE half, round-to-nearest, saturating rather than overflowing to inf. */
        private fun half(value: Float): Short {
            val bits = java.lang.Float.floatToRawIntBits(value)
            val sign = (bits ushr 16) and 0x8000
            val magnitude = bits and 0x7fffffff

            if (magnitude >= 0x7f800000) {
                // NaN keeps a payload bit so it stays a NaN; infinity stays infinity.
                val nan = if (magnitude > 0x7f800000) 0x200 else 0
                return (sign or 0x7c00 or nan).toShort()
            }

            val rounded = magnitude + 0x1000
            if (rounded >= 0x47800000) return (sign or 0x7bff).toShort()   // saturate to 65504
            if (rounded >= 0x38800000) return (sign or ((rounded - 0x38000000) ushr 13)).toShort()
            if (magnitude < 0x33000000) return sign.toShort()              // rounds to zero

            // Subnormal: shift the implicit one back in by hand.
            val exponent = magnitude ushr 23
            val shift = 126 - exponent
            val mantissa = (magnitude and 0x7fffff) or 0x800000
            return (sign or ((mantissa + (1 shl (shift - 1))) ushr shift)).toShort()
        }

        // [view], [sampler] at bindings [first], [first]+1 of [group], and `apply_lut` over
        // straight-alpha colour.
        fun wgsl(group: Int, first: Int) = """
@group($group) @binding($first) var lut_tex: texture_3d<f32>;
@group($group) @binding(${first + 1}) var lut_sampler: sampler;

fn apply_lut(rgb: vec3<f32>) -> vec3<f32> {
    let lookup = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // Entry centres, so the ends of the table land on the ends of the range.
    let half_texel = 0.5 / f32(textureDimensions(lut_tex).x);
    let coord = mix(vec3<f32>(half_texel), vec3<f32>(1.0 - half_texel), lookup);
    // Level 0: callers may be in non-uniform control flow, where implicit derivatives are not
    // allowed.
    return textureSampleLevel(lut_tex, lut_sampler, coord, 0.0).rgb;
}
"""
    }
}
