package ca.mpreg.webgpuviewer.transition

import androidx.compose.ui.geometry.Offset
import androidx.webgpu.GPUCommandEncoder
import androidx.webgpu.GPUTexture
import ca.mpreg.webgpuviewer.viewer.ImagePage
import kotlin.math.cos
import kotlin.math.sin

object TransitionCubeOuter : Transition() {
    private const val HALF_PI = (Math.PI / 2.0).toFloat()
    private const val FOV = 4f
    private const val FACE_DEPTH = FOV / (FOV - 1f)
    private const val PHASE_IN_END = 0.1f
    private const val PHASE_OUT_START = 0.9f

    private fun mat4(
        m00: Float, m01: Float, m02: Float, m03: Float,
        m10: Float, m11: Float, m12: Float, m13: Float,
        m20: Float, m21: Float, m22: Float, m23: Float,
        m30: Float, m31: Float, m32: Float, m33: Float,
    ) = floatArrayOf(
        m00, m01, m02, m03,
        m10, m11, m12, m13,
        m20, m21, m22, m23,
        m30, m31, m32, m33,
    )

    private fun rotateY(angle: Float): FloatArray {
        val c = cos(angle)
        val s = sin(angle)
        return mat4(
            c, 0f, -s, 0f,
            0f, 1f, 0f, 0f,
            s, 0f, c, 0f,
            0f, 0f, 0f, 1f,
        )
    }

    private fun translate(x: Float, y: Float, z: Float) = mat4(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        x, y, z, 1f,
    )

    private fun scale(x: Float, y: Float, z: Float) = mat4(
        x, 0f, 0f, 0f,
        0f, y, 0f, 0f,
        0f, 0f, z, 0f,
        0f, 0f, 0f, 1f,
    )

    private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val result = FloatArray(16)
        for (col in 0 until 4) {
            for (row in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[k * 4 + row] * b[col * 4 + k]
                }
                result[col * 4 + row] = sum
            }
        }
        return result
    }

    private fun buildFaceMatrix(
        rotAngle: Float,
        screenAspect: Float,
        imgWidth: Float,
        imgHeight: Float,
        isSide: Boolean,
        phase: Float,
    ): FloatArray {
        val d = FACE_DEPTH
        val pushBack = 5f * d
        // Scale face so it fills NDC ±1 at rest (matches flat image size, no zoom during phase-in)
        // ndc edge = s * FOV / (pushBack - s + FOV) = 1  →  s = (pushBack + FOV) / (FOV + 1)
        val s = (pushBack + FOV) / (FOV + 1f)
        val faceScaleMat = scale(s, (imgHeight / imgWidth) * screenAspect * s, 1f)
        val baseMat = if (isSide) {
            multiply(rotateY(HALF_PI), multiply(translate(0f, 0f, -s), faceScaleMat))
        } else {
            multiply(translate(0f, 0f, -s), faceScaleMat)
        }
        val mat = multiply(translate(0f, 0f, pushBack), multiply(rotateY(-rotAngle), baseMat))
        // Perspective projection: output.w = z + fov, ndc = xy / w
        val projMat = mat4(
            FOV, 0f, 0f, 0f,
            0f, FOV, 0f, 0f,
            0f, 0f, 1f, 1f,
            0f, 0f, 0f, FOV,
        )
        val result = multiply(projMat, mat)
        result[14] = phase
        return result
    }

    override fun render(
        page1: ImagePage,
        page2: ImagePage,
        encoder: GPUCommandEncoder,
        dst: GPUTexture,
        frac: Float,
        pos1: Offset,
        pos2: Offset,
    ) {
        val frac = -frac

        val t = if (frac > 0f) frac else 1f + frac

        val frontPhase = if (t < PHASE_IN_END) t / PHASE_IN_END else 1f
        val sidePhase =
            if (t > PHASE_OUT_START) 1f - (t - PHASE_OUT_START) / (1f - PHASE_OUT_START) else 1f

        val rotAngle = when {
            t < PHASE_IN_END -> 0f
            t > PHASE_OUT_START -> HALF_PI
            else -> (t - PHASE_IN_END) / (PHASE_OUT_START - PHASE_IN_END) * HALF_PI
        }

        val screenAspect = dst.width.toFloat() / dst.height.toFloat()

        val frontPage: ImagePage
        val sidePage: ImagePage
        if (frac > 0f) {
            frontPage = page1
            sidePage = page2
        } else {
            frontPage = page2
            sidePage = page1
        }

        val frontImg = frontPage.image
        val sideImg = sidePage.image

        val frontMat = buildFaceMatrix(
            rotAngle, screenAspect,
            frontImg?.width?.toFloat() ?: 1f,
            frontImg?.height?.toFloat() ?: 1f,
            isSide = false,
            phase = frontPhase,
        )
        val sideMat = buildFaceMatrix(
            rotAngle, screenAspect,
            sideImg?.width?.toFloat() ?: 1f,
            sideImg?.height?.toFloat() ?: 1f,
            isSide = true,
            phase = sidePhase,
        )

        if (t < 0.5f) {
            TransitionCube.render(sidePage, encoder, dst, sideMat)
            TransitionCube.render(frontPage, encoder, dst, frontMat)
        } else {
            TransitionCube.render(frontPage, encoder, dst, frontMat)
            TransitionCube.render(sidePage, encoder, dst, sideMat)
        }
    }
}
