package ca.mpreg.webgpuviewer

import android.content.Context
import android.util.AttributeSet
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.AbstractComposeView
import ca.mpreg.webgpuviewer.viewer.ImageViewer
import ca.mpreg.webgpuviewer.viewer.ImageViewerState

open class ImageView(context: Context, attrs: AttributeSet? = null, isVertical: Boolean = false) :
    AbstractComposeView(context, attrs) {

    constructor(context: Context, attrs: AttributeSet? = null) : this(
        context, attrs, context.obtainStyledAttributes(
            attrs, intArrayOf(android.R.attr.orientation)
        ).let {
            val orientation = it.getInt(0, 0)
            it.recycle()
            orientation == 1
        })

    open val state: ImageViewerState = ImageViewerState(isVertical)

    @Composable
    override fun Content() {
        ImageViewer(state = state)
    }
}
