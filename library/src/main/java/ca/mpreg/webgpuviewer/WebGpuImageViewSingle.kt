package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView

class WebGpuImageViewSingle(context: Context, attrs: AttributeSet? = null) :
    AbstractComposeView(context, attrs) {
    var state: WebGpuImageViewerSingleState = WebGpuImageViewerSingleState()

    @Composable
    override fun Content() {
        WebGpuImageViewerSingle(state = state)
    }
}
