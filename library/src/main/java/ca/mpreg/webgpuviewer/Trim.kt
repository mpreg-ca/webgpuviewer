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
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexture
import androidx.webgpu.MapMode
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.time.Duration.Companion.milliseconds

class Trim {
    companion object {
        val device get() = WebGpuRenderer.device
        val dispatcher get() = WebGpuRenderer.dispatcher
        val instance get() = WebGpuRenderer.instance

        var pipelineAll: GPUComputePipeline
        var pipelineLeft: GPUComputePipeline
        var pipelineTop: GPUComputePipeline
        var pipelineRight: GPUComputePipeline
        var pipelineBottom: GPUComputePipeline

        init {
            pipelineAll = device.createComputePipeline(
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

            val module = device.createShaderModule(
                GPUShaderModuleDescriptor(
                    shaderSourceWGSL = GPUShaderSourceWGSL(TrimShaderSingle)
                )
            )

            pipelineLeft = device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = module, entryPoint = "find_left")
                )
            )

            pipelineRight = device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = module, entryPoint = "find_right")
                )
            )

            pipelineTop = device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = module, entryPoint = "find_top")
                )
            )

            pipelineBottom = device.createComputePipeline(
                GPUComputePipelineDescriptor(
                    GPUComputeState(module = module, entryPoint = "find_bottom")
                )
            )
        }

        suspend fun find(image: Image, r: Float, g: Float, b: Float, threshold: Float): Rect {
            return withContext(dispatcher) {
                val mipmap = image.mipmaps[0]

                if (mipmap.tilesCols == 1 && mipmap.tilesRows == 1) {
                    val res = find(mipmap.textures[0], pipelineAll, r, g, b, threshold)

                    val job = launch {
                        while (true) {
                            instance.processEvents()
                            delay(1.milliseconds)
                        }
                    }

                    try {
                        return@withContext res.await()
                    } finally {
                        job.cancel()
                    }
                }

                val left = mutableListOf<Deferred<Rect>>()
                val right = mutableListOf<Deferred<Rect>>()
                val top = mutableListOf<Deferred<Rect>>()
                val bottom = mutableListOf<Deferred<Rect>>()

                for (row in 0 until mipmap.tilesRows) {
                    left.add(
                        find(
                            mipmap.textures[row * mipmap.tilesCols],
                            pipelineLeft,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                    right.add(
                        find(
                            mipmap.textures[row * mipmap.tilesCols + mipmap.tilesCols - 1],
                            pipelineRight,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }

                for (col in 0 until mipmap.tilesCols) {
                    top.add(find(mipmap.textures[col], pipelineTop, r, g, b, threshold))
                    bottom.add(
                        find(
                            mipmap.textures[col + (mipmap.tilesCols - 1) * mipmap.tilesRows],
                            pipelineBottom,
                            r,
                            g,
                            b,
                            threshold
                        )
                    )
                }

                val job = launch {
                    while (true) {
                        instance.processEvents()
                        delay(1.milliseconds)
                    }
                }

                try {
                    return@withContext Rect(
                        left.awaitAll().map({ it.left }).min(),
                        top.awaitAll().map({ it.top }).min(),
                        right.awaitAll()
                            .map({ it.right + mipmap.tilesize * (mipmap.tilesCols - 1) }).max(),
                        bottom.awaitAll()
                            .map({ it.bottom + mipmap.tilesize * (mipmap.tilesRows - 1) }).max(),
                    )
                } finally {
                    job.cancel()
                }
            }
        }

        fun find(
            texture: GPUTexture,
            pipeline: GPUComputePipeline,
            r: Float,
            g: Float,
            b: Float,
            threshold: Float
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
            stagingBuffer.unmap()

            val res = CompletableDeferred<Rect>()

            stagingBuffer.mapAsync(
                MapMode.Read,
                0,
                16,
                { it.run() },
                object : GPURequestCallback<Unit> {
                    override fun onResult(result: Unit) {
                        val output = stagingBuffer.getConstMappedRange()
                        output.order(ByteOrder.nativeOrder())
                        stagingBuffer.unmap()
                        res.complete(
                            Rect(
                                output.getInt(0),
                                output.getInt(4),
                                output.getInt(8),
                                output.getInt(12),
                            )
                        )

                        uniformBuffer.destroy()
                        resultBuffer.destroy()
                        stagingBuffer.destroy()
                    }

                    override fun onError(exception: Exception) {
                        Log.e("WebGpuRenderer", "Error in trim", exception)
                    }
                })

            return res
        }
    }
}