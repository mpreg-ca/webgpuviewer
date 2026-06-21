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
        val frames = mutableListOf<Image>()
        val durations = mutableListOf<Long>()
        do {
            val res = dec.decodeNext()
            val image = Image(res.image, res.width, res.height)
            frames.add(image)
            durations.add(res.duration.toLong())
        } while (dec.page != 0)

        binding.composeView1.apply {
            layoutParams.width = width
            layoutParams.height = height
            renderer.postInit = {
                renderer.addImage(frames[0])
                renderer.fitScale = renderer.minScale
                renderer.scale = renderer.fitScale
                renderer.render()
            }
            renderer.postInit()
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
