package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView

class WebGpuImageView(context: Context, attrs: AttributeSet? = null) :
    AbstractComposeView(context, attrs) {
    var renderer: WebGpuRenderer = WebGpuRenderer()

    @Composable
    override fun Content() {
        WebGpuImageViewer(renderer = renderer)
    }
}
