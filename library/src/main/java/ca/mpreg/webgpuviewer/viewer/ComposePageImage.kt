package ca.mpreg.webgpuviewer.viewer

import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.Canvas
import android.view.View
import android.view.View.MeasureSpec
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.findViewTreeLifecycleOwner
import androidx.lifecycle.findViewTreeViewModelStoreOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.findViewTreeSavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import ca.mpreg.webgpuviewer.renderer.Image
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.android.awaitFrame
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Turns a composable into an [Image] the viewer blits like any decoded page.
 *
 * Composed over the surface, chrome sits on its own layer and clock and slides out of step during a swipe;
 * in a page texture it stays in the render pass and rides every transition, spread and pan.
 *
 * One-shot: [render] draws the first laid-out frame and drops the view, so content must be re-rendered on
 * change rather than animate on its own.
 */
class ComposePageImage(private val host: ViewGroup) {

    suspend fun render(
        width: Int,
        height: Int,
        backgroundColor: Int,
        content: @Composable () -> Unit,
    ): Image {
        require(width > 0 && height > 0) { "Compose page must have a size: ${width}x$height" }

        val pixels =
            withContext(Dispatchers.Main) { renderPixels(width, height, backgroundColor, content) }

        val image = Image(width, height)
        try {
            image.backgroundColor = backgroundColor
            image.useImageLut = false
            WebGpuRenderer.withContext { _ -> image.mipmaps[0].update(pixels) }
        } catch (e: Throwable) {
            image.cleanup()
            throw e
        }
        return image
    }

    suspend fun renderPixels(
        width: Int,
        height: Int,
        backgroundColor: Int,
        content: @Composable () -> Unit,
    ): ByteBuffer {
        val view = ComposeView(host.context).apply {
            // The host outlives a single page, so the composition has to go when this view does.
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnDetachedFromWindowOrReleasedFromPool)
            // Laid out and drawn by hand below, never by the host: INVISIBLE keeps it out of the
            // window's own draw and away from touch, while still measuring and composing.
            visibility = View.INVISIBLE
            layoutParams = FrameLayout.LayoutParams(width, height)
            // Compose reads these off the view tree, which carries them only if the host activity installed them;
            // set here so the page composes under any host.
            setViewTreeLifecycleOwner(
                host.findViewTreeLifecycleOwner() ?: host.context.owner<LifecycleOwner>()
            )
            setViewTreeSavedStateRegistryOwner(
                host.findViewTreeSavedStateRegistryOwner()
                    ?: host.context.owner<SavedStateRegistryOwner>(),
            )
            setViewTreeViewModelStoreOwner(
                host.findViewTreeViewModelStoreOwner() ?: host.context.owner<ViewModelStoreOwner>(),
            )
            setContent(content)
        }

        host.addView(view)
        try {
            // Composition lands the frame after attach and its layout the one after; measuring earlier can't pull
            // it forward, so wait the frames out.
            repeat(COMPOSE_FRAMES) { awaitFrame() }

            view.measure(
                MeasureSpec.makeMeasureSpec(width, MeasureSpec.EXACTLY),
                MeasureSpec.makeMeasureSpec(height, MeasureSpec.EXACTLY),
            )
            view.layout(0, 0, width, height)

            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            try {
                val canvas = Canvas(bitmap)
                canvas.drawColor(backgroundColor)
                view.draw(canvas)

                // ARGB_8888 bytes are already RGBA8Unorm order; writeTexture only needs a direct, native-ordered buffer.
                val buffer = ByteBuffer.allocateDirect(width * height * Int.SIZE_BYTES)
                    .order(ByteOrder.nativeOrder())
                bitmap.copyPixelsToBuffer(buffer)
                buffer.rewind()
                return buffer
            } finally {
                bitmap.recycle()
            }
        } finally {
            // Also disposes the composition, via the strategy set above.
            host.removeView(view)
        }
    }

    private companion object {
        const val COMPOSE_FRAMES = 2

        /** The activity behind a view's context, which is every owner a composition asks for. */
        inline fun <reified T> Context.owner(): T? {
            var context: Context? = this
            while (context != null) {
                if (context is T) return context
                context = (context as? ContextWrapper)?.baseContext
            }
            return null
        }
    }
}
