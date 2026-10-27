package ca.mpreg.webgpuviewer

import android.graphics.Rect
import android.util.Log
import java.nio.ByteBuffer

class Trim {
    companion object {
        private const val TAG = "Trim"

        /**
         * For each of [colors], in input order, the bounding box of the pixels not within
         * [threshold] of it.
         *
         * [pixels] must be a direct buffer of tightly packed RGBA8, `width * height * 4` bytes.
         * Falls back to full bounds if the native pass can't read it.
         */
        fun findAllCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            colors: List<FloatArray>,
            threshold: Float
        ): List<Rect> {
            require(colors.isNotEmpty()) { "colors must not be empty" }
            require(colors.all { it.size >= 3 }) { "each color must have at least 3 elements [r, g, b]" }

            val flat = FloatArray(colors.size * 3)
            colors.forEachIndexed { i, color ->
                flat[i * 3] = color[0]
                flat[i * 3 + 1] = color[1]
                flat[i * 3 + 2] = color[2]
            }

            val bounds = IntArray(colors.size * 4)
            if (!TrimNative.findTrim(pixels, width, height, flat, threshold, bounds)) {
                Log.w(TAG, "findAllCpu: native trim rejected ${width}x$height, using full bounds")
                return List(colors.size) { Rect(0, 0, width, height) }
            }

            // Native bounds are inclusive maxima.
            return List(colors.size) { i ->
                Rect(
                    bounds[i * 4],
                    bounds[i * 4 + 1],
                    (bounds[i * 4 + 2] + 1).coerceAtMost(width),
                    (bounds[i * 4 + 3] + 1).coerceAtMost(height),
                )
            }
        }

        fun findCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            colors: List<FloatArray>,
            threshold: Float
        ): Rect = findAllCpu(pixels, width, height, colors, threshold)
            .minByOrNull { it.width().toLong() * it.height() } ?: Rect(0, 0, width, height)

        /**
         * The background colour implied by the image edges, as 0xAARRGGBB, or opaque white when
         * no edge is a solid colour.
         */
        fun detectBackgroundCpu(
            pixels: ByteBuffer,
            width: Int,
            height: Int,
            threshold: Float = 0.05f
        ): Int {
            if (!pixels.isDirect) {
                Log.w(TAG, "detectBackgroundCpu: pixels not direct, defaulting to white")
                return 0xFFFFFFFF.toInt()
            }
            return TrimNative.detectBackground(pixels, width, height, threshold)
        }
    }
}
