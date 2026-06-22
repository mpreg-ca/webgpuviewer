package ca.mpreg.webgpuviewer.test

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import ca.mpreg.imagedecoder.ImageDecoder
import ca.mpreg.webgpuviewer.Image
import ca.mpreg.webgpuviewer.test.databinding.MainActivityBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: MainActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val width = resources.displayMetrics.widthPixels
        val height = resources.displayMetrics.heightPixels

        val stream = assets.open("ref.png")
        val dec = ImageDecoder.new(stream)
        val res = dec.decodeNext(getTrim = true)
        val image = Image(res.image, res.width, res.height)

        binding.composeView1.apply {
            layoutParams.width = width
            layoutParams.height = height
            renderer.dpi = resources.displayMetrics.densityDpi / 100f
            renderer.post {
                renderer.images.add(image)

                renderer.homeScale = renderer.getMinScale(res.trim_width, res.trim_height)
                renderer.homeX = renderer.getHomeX(
                    -((res.trim_left.toFloat() - 0.5f * res.width) / res.trim_width.toFloat() + 0.5f),
                    renderer.homeScale
                )
                renderer.homeY = renderer.getHomeY(
                    -((res.trim_top.toFloat() - 0.5f * res.height) / res.trim_height.toFloat() + 0.5f),
                    renderer.homeScale
                )

                renderer.render()
                renderer.home()
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
