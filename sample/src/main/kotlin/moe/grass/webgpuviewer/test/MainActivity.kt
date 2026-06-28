package ca.mpreg.webgpuviewer.test

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ca.mpreg.imagedecoder.ImageDecoder
import ca.mpreg.webgpuviewer.Image
import ca.mpreg.webgpuviewer.Trim
import ca.mpreg.webgpuviewer.WebGpuImageViewerPage
import ca.mpreg.webgpuviewer.test.databinding.MainActivityBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

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
            val stream = assets.open("ref.png")
            val dec = ImageDecoder.new(stream)
            Log.i("start", "start ${dec.pages}")
            val res = dec.decodeNext()

            binding.composeView1.state.apply {
                dpi = resources.displayMetrics.densityDpi / 100f

                val state = this
                fetchPage = { index ->
                    WebGpuImageViewerPage(Image(res.image, res.width, res.height)).apply {
                        trim = Trim.find(image, 1f, 1f, 1f, 10f / 255)

                        parent = state
                        x = homeX
                        y = homeY
                        scale = homeScale
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
