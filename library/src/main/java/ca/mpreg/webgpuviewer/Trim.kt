package ca.mpreg.webgpuviewer

import android.graphics.Rect
import android.util.Log
import androidx.webgpu.BufferUsage
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.MapMode
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.Mipmap
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.time.Duration.Companion.milliseconds

class Trim {
    companion object {
        private const val TAG = "Trim"

        val device get() = WebGpuRenderer.device
        val instance get() = WebGpuRenderer.instance

        val pipelineAll: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(
                        device.createShaderModule(
                            GPUShaderModuleDescriptor(
                                shaderSourceWGSL = GPUShaderSourceWGSL(TrimShader)
                            )
                        )
                    )
                )
            )
        }

        private val singleModule by lazy {
            device.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(TrimShaderSingle)
                )
            )
        }

        val pipelineLeft: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_left")
                )
            )
        }

        val pipelineRight: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_right")
                )
            )
        }

        val pipelineTop: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_top")
                )
            )
        }

        val pipelineBottom: GPUComputePipeline by lazy {
            device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = singleModule, entryPoint = "find_bottom")
                )
            )
        }

        /**
         * Find trim bounds. Call from within WebGpuRenderer.withContext for best performance.
         */
        suspend fun find(image: Image, r: Float, g: Float, b: Float, threshold: Float): Rect {
            return WebGpuRenderer.withContext { device ->
                findInContext(device, image, r, g, b, threshold)
            }
        }

        /**
         * Find trim bounds when already inside a GPU context. Avoids extra context switch.
         * Returns a Rect with 0,0,width,height if trim detection fails.
         */
        suspend fun findInContext(
            device: GPUDevice,
            image: Image,
            r: Float,
            g: Float,
            b: Float,
            threshold: Float
        ): Rect {
            if (image.mipmaps.isEmpty()) {
                Log.w(TAG, "findInContext: image has no mipmaps, returning full bounds")
                return Rect(0, 0, image.width, image.height)
            }

            val mipmap = image.mipmaps[0]

            if (mipmap.textures.isEmpty()) {
                Log.w(TAG, "findInContext: mipmap has no textures, returning full bounds")
                return Rect(0, 0, image.width, image.height)
            }

            return try {
                if (mipmap.tilesCols == 1 && mipmap.tilesRows == 1) {
                    findSingleTile(mipmap.textures[0], r, g, b, threshold)
                } else {
                    findMultiTile(mipmap, r, g, b, threshold)
                }
            } catch (e: Exception) {
                Log.e(TAG, "findInContext: error during trim detection, returning full bounds", e)
                Rect(0, 0, image.width, image.height)
            }
        }

        private suspend fun findSingleTile(
            texture: GPUTexture,
            r: Float, g: Float, b: Float, threshold: Float
        ): Rect = coroutineScope {
            val res = dispatchTrimCompute(texture, pipelineAll, r, g, b, threshold)

            val job = launch {
                while (true) {
                    instance.processEvents()
                    delay(1.milliseconds)
                }
            }

            try {
                res.await()
            } finally {
                job.cancel()
            }
        }

        private suspend fun findMultiTile(
            mipmap: Mipmap,
            r: Float, g: Float, b: Float, threshold: Float
        ): Rect = coroutineScope {
            val left = mutableListOf<Deferred<Rect>>()
            val right = mutableListOf<Deferred<Rect>>()
            val top = mutableListOf<Deferred<Rect>>()
            val bottom = mutableListOf<Deferred<Rect>>()

            for (row in 0 until mipmap.tilesRows) {
                val leftIdx = row * mipmap.tilesCols
                val rightIdx = row * mipmap.tilesCols + mipmap.tilesCols - 1

                if (leftIdx < mipmap.textures.size) {
                    left.add(
                        dispatchTrimCompute(
                            mipmap.textures[leftIdx],
                            pipelineLeft,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }
                if (rightIdx < mipmap.textures.size) {
                    right.add(
                        dispatchTrimCompute(
                            mipmap.textures[rightIdx],
                            pipelineRight,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }
            }

            for (col in 0 until mipmap.tilesCols) {
                val topIdx = col
                val bottomIdx = (mipmap.tilesRows - 1) * mipmap.tilesCols + col

                if (topIdx < mipmap.textures.size) {
                    top.add(
                        dispatchTrimCompute(
                            mipmap.textures[topIdx],
                            pipelineTop,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }
                if (bottomIdx < mipmap.textures.size) {
                    bottom.add(
                        dispatchTrimCompute(
                            mipmap.textures[bottomIdx],
                            pipelineBottom,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }
            }

            val job = launch {
                while (true) {
                    instance.processEvents()
                    delay(1.milliseconds)
                }
            }

            try {
                val leftResults = left.awaitAll()
                val topResults = top.awaitAll()
                val rightResults = right.awaitAll()
                val bottomResults = bottom.awaitAll()

                Rect(
                    leftResults.minOfOrNull { it.left } ?: 0,
                    topResults.minOfOrNull { it.top } ?: 0,
                    (rightResults.maxOfOrNull { it.right } ?: mipmap.tilesize) +
                            mipmap.tilesize * (mipmap.tilesCols - 1),
                    (bottomResults.maxOfOrNull { it.bottom } ?: mipmap.tilesize) +
                            mipmap.tilesize * (mipmap.tilesRows - 1),
                )
            } finally {
                job.cancel()
            }
        }

        /**
         * Dispatch a trim compute operation. Always creates fresh buffers to avoid
         * concurrency issues with buffer reuse during async operations.
         */
        private fun dispatchTrimCompute(
            texture: GPUTexture,
            pipeline: GPUComputePipeline,
            r: Float, g: Float, b: Float, threshold: Float
        ): Deferred<Rect> {
            val uniformBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16, usage = BufferUsage.Uniform or BufferUsage.CopyDst
                )
            )
            val resultBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16,
                    usage = BufferUsage.Storage or BufferUsage.CopySrc or BufferUsage.CopyDst
                )
            )
            val stagingBuffer = device.createBuffer(
                GPUBufferDescriptor(
                    size = 16, usage = BufferUsage.CopyDst or BufferUsage.MapRead
                )
            )

            val byteBuffer = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
            byteBuffer.putFloat(0, r)
            byteBuffer.putFloat(4, g)
            byteBuffer.putFloat(8, b)
            byteBuffer.putFloat(12, threshold)

            device.queue.writeBuffer(uniformBuffer, 0, byteBuffer)

            val initBuffer = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder())
            initBuffer.putInt(texture.width)
            initBuffer.putInt(texture.height)
            initBuffer.putInt(0)
            initBuffer.putInt(0)
            initBuffer.flip()

            device.queue.writeBuffer(resultBuffer, 0L, initBuffer)

            val encoder = device.createCommandEncoder()
            val pass = encoder.beginComputePass()
            pass.setPipeline(pipeline)
            pass.setBindGroup(
                0, device.createBindGroup(
                    GPUBindGroupDescriptor(
                        layout = pipeline.getBindGroupLayout(0), entries = arrayOf(
                            GPUBindGroupEntry(0, textureView = texture.createView()),
                            GPUBindGroupEntry(1, buffer = resultBuffer),
                            GPUBindGroupEntry(2, buffer = uniformBuffer),
                        )
                    )
                )
            )

            pass.dispatchWorkgroups(
                ceil(texture.width / 8.0).toInt(), ceil(texture.height / 8.0).toInt()
            )
            pass.end()

            encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 16)

            device.queue.submit(arrayOf(encoder.finish()))

            val res = CompletableDeferred<Rect>()

            stagingBuffer.mapAsync(
                MapMode.Read,
                0,
                16,
                { it.run() },
                object : GPURequestCallback<Unit> {
                    override fun onResult(result: Unit) {
                        try {
                            val output = stagingBuffer.getConstMappedRange()
                            output.order(ByteOrder.nativeOrder())
                            val rect = Rect(
                                output.getInt(0),
                                output.getInt(4),
                                (output.getInt(8) + 1).coerceAtMost(texture.width),
                                (output.getInt(12) + 1).coerceAtMost(texture.height),
                            )
                            stagingBuffer.unmap()
                            res.complete(rect)
                        } catch (e: Exception) {
                            Log.e(TAG, "Error reading trim result", e)
                            res.complete(Rect(0, 0, texture.width, texture.height))
                        } finally {
                            uniformBuffer.destroy()
                            resultBuffer.destroy()
                            stagingBuffer.destroy()
                        }
                    }

                    override fun onError(exception: Exception) {
                        Log.e(TAG, "Error in trim mapAsync", exception)
                        // Complete with full bounds instead of leaving deferred hanging
                        res.complete(Rect(0, 0, texture.width, texture.height))
                        uniformBuffer.destroy()
                        resultBuffer.destroy()
                        stagingBuffer.destroy()
                    }
                })

            return res
        }

        @Deprecated(
            "Use find() or findInContext() instead",
            ReplaceWith("find(image, r, g, b, threshold)")
        )
        fun find(
            texture: GPUTexture,
            pipeline: GPUComputePipeline,
            r: Float,
            g: Float,
            b: Float,
            threshold: Float
        ): Deferred<Rect> = dispatchTrimCompute(texture, pipeline, r, g, b, threshold)
    }
}
