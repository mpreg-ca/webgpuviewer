package ca.mpreg.webgpuviewer.test

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.spring
import androidx.compose.ui.platform.AndroidUiDispatcher
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.lifecycle.lifecycleScope
import ca.mpreg.webgpuviewer.WebGpuImageViewer
import ca.mpreg.webgpuviewer.WebGpuRenderer
import ca.mpreg.webgpuviewer.orZero
import ca.mpreg.webgpuviewer.test.databinding.MainActivityBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var binding: MainActivityBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = MainActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val stream = assets.open("ref.png")
        val bitmap = BitmapFactory.decodeStream(stream)

        val renderer = WebGpuRenderer()

        binding.composeView1.apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed)
            setContent {
                WebGpuImageViewer(renderer = renderer, bitmap = bitmap)
            }
        }

        binding.btnUp.setOnClickListener {
            renderer.animationJob = lifecycleScope.launch(AndroidUiDispatcher.Main) {
                val startY = renderer.y
                val max_y = renderer.maxY()
                val targetY = (startY + 0.1f).coerceIn(-max_y, max_y)
                animate(0f, 1f, animationSpec = spring()) { value, _ ->
                    renderer.y = (startY + (targetY - startY) * value).orZero()
                    renderer.render()
                }
            }
        }

        binding.btnDown.setOnClickListener {
            renderer.animationJob = lifecycleScope.launch(AndroidUiDispatcher.Main) {
                val startY = renderer.y
                val max_y = renderer.maxY()
                val targetY = (startY - 0.1f).coerceIn(-max_y, max_y)
                animate(0f, 1f, animationSpec = spring()) { value, _ ->
                    renderer.y = (startY + (targetY - startY) * value).orZero()
                    renderer.render()
                }
            }
        }
    }
}
