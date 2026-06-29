package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView
import ca.mpreg.webgpuviewer.viewer.WebGpuImageViewer
import ca.mpreg.webgpuviewer.viewer.WebGpuImageViewerState

class WebGpuImageView(context: Context, attrs: AttributeSet? = null, isVertical: Boolean = false) :
    AbstractComposeView(context, attrs) {
    var state: WebGpuImageViewerState = WebGpuImageViewerState(isVertical)

    @Composable
    override fun Content() {
        WebGpuImageViewer(state = state)
    }
}
