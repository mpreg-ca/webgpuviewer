package ca.mpreg.webgpuviewer.test

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import ca.mpreg.imagedecoder.ImageDecoder
import ca.mpreg.webgpuviewer.Image
import ca.mpreg.webgpuviewer.WebGpuImageViewerPage
import ca.mpreg.webgpuviewer.WebGpuRenderer
import ca.mpreg.webgpuviewer.test.databinding.MainActivityBinding
import ca.mpreg.webgpuviewer.transitions.TransitionBasic
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var binding: MainActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val width = resources.displayMetrics.widthPixels
        val height = resources.displayMetrics.heightPixels

        binding.composeView1.apply {
            layoutParams.width = width
            layoutParams.height = height
        }

        CoroutineScope(Dispatchers.Default).launch {
            val page1 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    WebGpuImageViewerPage(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView1.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }

            val page2 = withContext(Dispatchers.Default) {
                val stream = assets.open("ref.png")
                val dec = ImageDecoder.new(stream)
                dec.decodeNext()
            }.let {
                withContext(WebGpuRenderer.dispatcher) {
                    WebGpuImageViewerPage(Image(it.image, it.width, it.height)).apply {
                        parent = binding.composeView1.state
                        x = homeX
                        y = homeY
                        scale = homeScale
                    }
                }
            }

            binding.composeView1.state.apply {
                dpi = resources.displayMetrics.densityDpi / 100f

                isVertical = true
                transition = TransitionBasic.Vertical::render

                haveNext = true
                havePrev = true

                fetchPage = { index ->
                    if (index == 0) {
                        page1
                    } else {
                        page2
                    }
                }

                post {
                    render()
                }
            }
        }

//        CoroutineScope(Dispatchers.Default).launch {
//            var frame = 0
//            while(true) {
//                val elapsed = measureTime {
//                    withContext(Dispatchers.Main) {
//                        binding.composeView1.renderer.let { r ->
//                            r.images.clear()
//                            r.images.add(frames[frame])
//                            r.render()
//                        }
//                    }
//                }
//
//                delay((durations[frame] - elapsed.inWholeMilliseconds).coerceAtLeast(0).milliseconds)
//                frame = (frame + 1) % frames.size
//            }
//        }
    }
}
