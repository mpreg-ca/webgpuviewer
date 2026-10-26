package ca.mpreg.webgpuviewer.viewer

import android.content.Context
import android.content.ContextWrapper
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.PorterDuff
import android.util.Log
import android.view.Choreographer
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionContext
import androidx.compose.runtime.rememberCompositionContext
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import ca.mpreg.webgpuviewer.renderer.WebGpuRenderer
import ca.mpreg.webgpuviewer.viewer.ComposePageHost.Companion.IDLE_INTERVAL_MS
import ca.mpreg.webgpuviewer.viewer.ComposePageHost.Companion.STILL_PASSES
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.min

/**
 * Hosts the compositions behind [ImagePage.Compose] pages; stacked over the surface so gestures reach
 * pages first (see [ComposePageHost.dispatchTouchEvent]). Page compositions are children of this one,
 * so CompositionLocals set around the viewer reach page content.
 */
@Composable
internal fun ComposePageHostView(state: ImageViewerState, modifier: Modifier = Modifier) {
    val scope = rememberCoroutineScope()
    val parentComposition = rememberCompositionContext()
    AndroidView(
        factory = { context -> ComposePageHost(context) },
        modifier = modifier,
        update = { it.bind(state, scope, parentComposition) },
        onRelease = { it.release() },
    )
}

/**
 * Holds the views behind [ImagePage.Compose] pages: measures, lays out and feeds them input, but keeps
 * their pixels off the window.
 *
 * A window-drawn view lands on the app's compositor layer while pages land on the viewer's surface:
 * two layers on two clocks, so chrome slides out of step during a swipe. Here the view is really
 * attached (recomposition, animation and input work normally) but [dispatchDraw] does nothing; its
 * pixels reach the screen only via the captured texture, in the page's render pass.
 *
 * The window's traversal can't drive it: the surface renders outside HWUI, so the window can idle for
 * minutes and pre-draw listeners never fire. Nor can its draw: `View.invalidate` drops invalidations
 * from a never-drawn view, so a skipped (or zero-alpha) subtree goes dark after its first frame.
 * Hence a private [Choreographer] callback, and each capture draws the content itself, setting
 * `PFLAG_DRAWN` so invalidations keep reaching [onDescendantInvalidated].
 *
 * Pages are found, not registered: [reconcile] builds a [PageRenderer] per in-reach
 * [ImagePage.Compose], so a page holds no reference to this host and can't be composed before its
 * builder finishes.
 */
internal class ComposePageHost(context: Context) : FrameLayout(context) {

    private var state: ImageViewerState? = null
    private var scope: CoroutineScope? = null
    private var parentComposition: CompositionContext? = null

    /** Main thread only; the pump reconciles it. */
    private val renderers = LinkedHashMap<ImagePage.Compose, PageRenderer>()

    /** Page that took the current gesture's down; receives the rest of it. */
    private var owner: PageRenderer? = null

    /** The down's map, reused for the gesture: a settling animation must not shift what a drag points at. */
    private var gestureMap: Matrix? = null

    init {
        // AndroidComposeView resolves these from the view tree, which has them only if the activity
        // installed them; setting them here lets a page compose under any host.
        setViewTreeLifecycleOwner(context.owner<LifecycleOwner>())
        setViewTreeSavedStateRegistryOwner(context.owner<SavedStateRegistryOwner>())
        setViewTreeViewModelStoreOwner(context.owner<ViewModelStoreOwner>())
    }

    private var pumping = false

    /** Pending callback is next frame's, so a wake-up has nothing to pull forward. */
    private var pendingImmediate = false

    /** Pages in reach may have changed; set by [wake], bind, layout and each idle poll. */
    private var reachDirty = true

    /** Every-frame passes since the last [reconcile], for [REACH_BACKSTOP_PASSES]. */
    private var passesSinceReconcile = 0

    private val onFrame = Choreographer.FrameCallback {
        // Backstop for unannounced changes, e.g. a neighbour replaced while the surface skips frames.
        if (!pendingImmediate || ++passesSinceReconcile >= REACH_BACKSTOP_PASSES) reachDirty = true
        pumping = false
        pendingImmediate = false
        if (reachDirty) {
            reachDirty = false
            passesSinceReconcile = 0
            reconcile()
        }
        placePages()
        // Per page, so one throwing (e.g. device lost) costs only its own frame. [renderers] isn't modified here.
        for (renderer in renderers.values) {
            try {
                renderer.captureIfDirty()
            } catch (e: Throwable) {
                renderer.failed(e)
            }
        }
        pump()
    }

    fun bind(
        state: ImageViewerState,
        scope: CoroutineScope,
        parentComposition: CompositionContext,
    ) {
        if (this.state !== state) {
            this.state?.let {
                if (it.onReachChanged === wake) it.onReachChanged = null
                it.coveredBy = null
            }
            state.onReachChanged = wake
        }
        // Renderers built on the old scope/composition would upload to a cancelled scope under a dead parent.
        if (this.scope !== scope || this.parentComposition !== parentComposition) {
            renderers.values.forEach { it.release() }
            renderers.clear()
        }
        this.state = state
        this.scope = scope
        this.parentComposition = parentComposition
        reachDirty = true
        pump(immediate = true)
    }

    /**
     * Re-homes [page], or the spread holding it, after a resize: it was placed on attach, unsized,
     * and would stay there until a touch clamped it. Not while zoomed or animating.
     */
    private fun rehome(page: ImagePage.Compose) {
        val state = state ?: return
        for (offset in -1..1) {
            val holder = state.getPage(offset) ?: continue
            val holds = holder === page ||
                    (holder is ImagePage.ImageSpread && (holder.left === page || holder.right === page))
            if (!holds || holder.animationJob?.isActive == true || !holder.atHomeScale) continue
            holder.setPos(holder.homeX, holder.homeY, holder.homeScale)
        }
    }

    /** [ImageViewerState.onReachChanged]: a page may have come within reach. */
    private val wake: () -> Unit = {
        post {
            reachDirty = true
            pump(immediate = true)
        }
    }

    fun release() {
        state?.let {
            if (it.onReachChanged === wake) it.onReachChanged = null
            it.coveredBy = null
        }
        Choreographer.getInstance().removeFrameCallback(onFrame)
        pumping = false
        renderers.values.toList().forEach { it.release() }
        renderers.clear()
        owner = null
        gestureMap = null
        state = null
        scope = null
        parentComposition = null
    }

    /**
     * Syncs [renderers] with the pages in reach: current page plus two neighbours, so a turn never lands
     * on an undrawn page. Out-of-reach pages are released (view, bitmap, texture), not kept for return.
     */
    private fun reconcile() {
        val state = state ?: return
        val scope = scope ?: return
        val parentComposition = parentComposition ?: return

        // A zoomed-out continuous viewer shows as many pages as fit, so reach follows the last frame.
        val below = if (state is ImageViewerContinuousState) state.pagesBelow + 1 else 1
        val wanted = LinkedHashMap<ImagePage.Compose, Boolean>()
        for (offset in -1..below) state.getPage(offset)?.collectComposePages(wanted, spread = false)

        renderers.keys.filter { it !in wanted }.forEach { page ->
            if (state.coveredBy === page) state.coveredBy = null
            renderers.remove(page)?.release()
        }
        wanted.forEach { (page, spread) ->
            if (page.destroyed) return@forEach
            renderers.getOrPut(page) { PageRenderer(page, scope, parentComposition) }
                .isSpreadSide = spread
        }
    }

    /**
     * Decides which pages the window carries this pass, and where: only still pages (see [pagedPlacement],
     * [continuousPlacement]); others hand over to their snapshot, moved by the surface on its own clock.
     * Touches map through the same placement.
     */
    private fun placePages() {
        val state = state ?: return

        renderers.forEach { (page, renderer) ->
            val was = renderer.placement
            renderer.placement = when {
                page.destroyed -> null
                state is ImageViewerContinuousState -> continuousPlacement(state, renderer)
                else -> pagedPlacement(state, page)
            }
            // Only tells the capture pass which pages the window has; both are drawn either way.
            page.presenting = renderer.placement == null
            if (renderer.hasView) renderer.placeView()

            // Before the page moves on its own, so the surface draws the frame that takes it over.
            if (!covers(state, page, renderer) && state.coveredBy === page) state.coveredBy = null

            if (was != renderer.placement) {
                invalidate()
                // Both sides change at a handover; without a surface redraw the snapshot lingers a frame.
                if ((was == null) != (renderer.placement == null)) page.parent?.invalidate()
            }
        }
    }

    /**
     * Current page of a paged viewer, or a spread side, while not mid-turn: transition shaders can't be
     * followed here. A spread side uses the spread's transform; see [ImagePage.ImageSpread.sideTransform].
     *
     * Zoom/pan stay here: their animations run on the main thread, so the transform is read in the frame
     * that sets it, and the page stays sharp where the 1:1 snapshot would be magnified.
     */
    private fun pagedPlacement(state: ImageViewerState, page: ImagePage.Compose): PagePlacement? {
        if (abs(state.pageOffset) >= SETTLED) return null
        val viewportWidth = state.width.toFloat()
        val current = state.getPage(0)
        val transform = when {
            page === current -> floatArrayOf(page.x, page.y, page.scale)
            current is ImagePage.ImageSpread -> current.sideTransform(page, viewportWidth)
                ?: return null

            else -> return null
        }
        return pagePlacement(
            viewportWidth = viewportWidth,
            viewportHeight = state.height.toFloat(),
            pageWidth = page.width,
            pageHeight = page.height,
            x = transform[0],
            y = transform[1],
            scale = transform[2],
            offsetX = WebGpuRenderer.offsetX,
            offsetY = WebGpuRenderer.offsetY,
        )
    }

    /**
     * Any on-screen continuous page that hasn't moved since the last pass; a moving page must be the
     * surface's (two layers, two clocks). Stillness is measured, not read off the gesture: a drag marks
     * panning from first touch (handing over on every tap), and a decode above can move a page with no
     * gesture.
     *
     * Costs [STILL_PASSES] of lag as motion stops, none as it starts: a scroll lands before the pass reads it.
     */
    private fun continuousPlacement(
        state: ImageViewerContinuousState,
        renderer: PageRenderer,
    ): PagePlacement? {
        val page = renderer.page
        val seen = if (page.isOnScreen && page.height > 0) state.screenPlacement(page) else null
        renderer.stillPasses = if (seen != null && seen == renderer.lastSeen) {
            renderer.stillPasses + 1
        } else {
            0
        }
        renderer.lastSeen = seen
        // A drag delivers no movement on some frames, and a handover on each of those would only
        // be handed back on the next.
        return seen.takeIf { renderer.stillPasses >= STILL_PASSES }
    }

    /** A spread side may be a composition and gets half the viewport, so it's tracked, not inferred. */
    private fun ImagePage.collectComposePages(
        into: MutableMap<ImagePage.Compose, Boolean>,
        spread: Boolean,
    ) {
        when (this) {
            is ImagePage.Compose -> into[this] = spread
            is ImagePage.ImageSpread -> {
                left?.collectComposePages(into, spread = true)
                right?.collectComposePages(into, spread = true)
            }

            else -> Unit
        }
    }

    /**
     * Runs the capture pass while bound: every frame while a page tracks its content, else every
     * [IDLE_INTERVAL_MS]. Can't stop: page turns happen outside HWUI, so only [reconcile] notices a
     * composition coming into reach; but refresh-rate polling would keep the main thread awake all session.
     */
    private fun pump(immediate: Boolean = false) {
        if (state == null || !isAttachedToWindow) return
        val next = immediate || renderers.values.any { it.needsEveryFrame() }
        if (pumping) {
            // Already waiting; only touch it to bring a polling callback forward.
            if (!next || pendingImmediate) return
            Choreographer.getInstance().removeFrameCallback(onFrame)
        }
        pumping = true
        pendingImmediate = next
        Choreographer.getInstance()
            .postFrameCallbackDelayed(onFrame, if (next) 0L else IDLE_INTERVAL_MS)
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        pump()
    }

    override fun onDetachedFromWindow() {
        Choreographer.getInstance().removeFrameCallback(onFrame)
        pumping = false
        // Nothing draws over the surface now, so it has to draw again.
        state?.coveredBy = null
        super.onDetachedFromWindow()
    }

    override fun onVisibilityChanged(changedView: View, visibility: Int) {
        super.onVisibilityChanged(changedView, visibility)
        if (!isShown) state?.coveredBy = null
    }

    /**
     * Draws only the still pages the window carries, at the place and scale the render pass would use, so
     * they animate at window rate with no capture. Moving pages are drawn by the render pass from their
     * snapshot. Exactly one is on screen: [ImagePage.Compose.currentImage] is null while this has it.
     */
    override fun dispatchDraw(canvas: Canvas) {
        val state = state
        var covering: ImagePage.Compose? = null
        renderers.values.forEach { renderer ->
            val placement = renderer.placement ?: return@forEach
            if (!renderer.hasView) return@forEach
            if (state != null && covers(state, renderer.page, renderer)) {
                // The whole viewport, as the surface letterboxes it, so the surface may stop
                // drawing underneath - see [ImageViewerState.coveredBy].
                canvas.drawColor(renderer.page.backgroundColor!!)
                covering = renderer.page
            }
            val saved = canvas.save()
            // Opaque like the snapshot under it, or antialiased edges blend over their own copy and the
            // page looks heavier still than moving.
            renderer.page.backgroundColor?.let { background ->
                canvas.clipRect(
                    placement.left,
                    placement.top,
                    placement.left + renderer.page.width * placement.scale,
                    placement.top + renderer.page.height * placement.scale,
                )
                canvas.drawColor(background)
            }
            // Placed by its own transform - see [PageRenderer.placeView].
            drawChild(canvas, renderer.view, drawingTime)
            canvas.restoreToCount(saved)
        }
        // Only once drawn: until then the surface's frame is all there is.
        if (state != null && covering != null) state.coveredBy = covering
    }

    /**
     * Whether the window draws [page] in place of the whole surface: only a full paged-viewer page, not a
     * spread side or strip page.
     */
    private fun covers(
        state: ImageViewerState,
        page: ImagePage.Compose,
        renderer: PageRenderer
    ): Boolean =
        state !is ImageViewerContinuousState && renderer.placement != null && !page.destroyed &&
                page.backgroundColor != null && page === state.getPage(0)

    /** Set while [PageRenderer.placeView] moves a view, which invalidates it without changing its content. */
    private var placingView = false

    /** Compose invalidations mark a page dirty; without this every page would re-upload every frame. */
    override fun onDescendantInvalidated(child: View, target: View) {
        super.onDescendantInvalidated(child, target)
        // A placement moved the view; its content is unchanged.
        if (placingView) return
        val renderer = renderers.values.firstOrNull { it.hasView && it.view === child } ?: return
        renderer.markDirty()
        // Drawn by the window while it is placed, so the window has to be told as well.
        if (renderer.placement != null) invalidate()
        pump(immediate = true)
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        super.onLayout(changed, left, top, right, bottom)
        if (!changed) return
        renderers.values.forEach { it.markDirty() }
        reachDirty = true
        pump(immediate = true)
    }

    /**
     * Routes a gesture to at most one page through its live pan/zoom transform. Content is laid out at the
     * origin at page size, so events are mapped to page space; the content decides by consuming the down.
     */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN) {
            owner = null
            gestureMap = null
            val refused = ArrayList<Pair<PageRenderer, Matrix>>(2)
            // Topmost first: a later page is drawn over an earlier one, so it gets first refusal.
            for (renderer in renderers.values.toList().asReversed()) {
                if (!renderer.acceptsInput()) continue
                val map = renderer.screenToPage() ?: continue
                if (!through(map, ev) { renderer.view.dispatchTouchEvent(it) }) {
                    refused.add(renderer to map)
                    continue
                }

                owner = renderer
                gestureMap = map
                // Nothing above may steal a gesture the content has already claimed.
                parent?.requestDisallowInterceptTouchEvent(true)
                break
            }
            // After the walk, so a refusal can't reach a page before the one above it saw the down.
            refused.forEach { (renderer, map) -> cancel(renderer, map, ev) }
            state?.contentClaimedTouch = owner != null
            return owner != null
        }

        val target = owner ?: return false
        val map = gestureMap ?: return false
        val handled = through(map, ev) { target.view.dispatchTouchEvent(it) }
        when (ev.actionMasked) {
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                owner = null
                gestureMap = null
            }
        }
        return handled
    }

    /** Scroll wheels, styluses etc., routed like [dispatchTouchEvent]; to the pager when no page takes it. */
    override fun dispatchGenericMotionEvent(ev: MotionEvent): Boolean {
        for (renderer in renderers.values.toList().asReversed()) {
            if (!renderer.acceptsInput()) continue
            val map = renderer.screenToPage() ?: continue
            if (through(map, ev) { renderer.view.dispatchGenericMotionEvent(it) }) return true
        }
        return false
    }

    /**
     * Cancels the gesture for [renderer]: a page that refused the down still received it, and the rest
     * goes to the pager, so without this it never sees an up or cancel.
     */
    private fun cancel(renderer: PageRenderer, map: Matrix, ev: MotionEvent) {
        val copy = MotionEvent.obtain(ev)
        try {
            copy.action = MotionEvent.ACTION_CANCEL
            copy.transform(map)
            renderer.view.dispatchTouchEvent(copy)
        } finally {
            copy.recycle()
        }
    }

    /** Sends a copy of [ev] in page coordinates; the original is left untouched. */
    private inline fun through(
        map: Matrix,
        ev: MotionEvent,
        send: (MotionEvent) -> Boolean
    ): Boolean {
        val copy = MotionEvent.obtain(ev)
        return try {
            copy.transform(map)
            send(copy)
        } finally {
            copy.recycle()
        }
    }

    /**
     * One page's composition view and its captures into the page's texture. The page owns the texture
     * (see [ImagePage.Compose.capture]); bitmap and buffer are reused because an animating page captures
     * every frame.
     */
    private inner class PageRenderer(
        val page: ImagePage.Compose,
        private val scope: CoroutineScope,
        private val parentComposition: CompositionContext,
    ) {
        private var built = false

        /** Whether [view] was built, so release never constructs it. */
        val hasView: Boolean get() = built

        /** Half the viewport, when this is a spread side. */
        var isSpreadSide: Boolean = false

        /** Where the window is drawing this page, or null while the render pass has it. */
        var placement: PagePlacement? = null

        /** Where the last pass found this page, carried or not - see [continuousPlacement]. */
        var lastSeen: PagePlacement? = null

        /** Consecutive passes [lastSeen] has held, so a pause mid-drag is not taken for rest. */
        var stillPasses = 0

        val view: ComposeView by lazy(LazyThreadSafetyMode.NONE) {
            ComposeView(this@ComposePageHost.context).apply {
                // Composes under the viewer's composition, inheriting locals (e.g. theme) set around the viewer.
                setParentCompositionContext(parentComposition)
                // The host outlives any one page, so the composition has to go when this view does.
                setViewCompositionStrategy(
                    ViewCompositionStrategy.DisposeOnDetachedFromWindowOrReleasedFromPool
                )
                setContent {
                    // Chrome sits in a page-sized slot; centring here saves every content doing it.
                    Box(contentAlignment = Alignment.Center) { with(page) { Content() } }
                }
                built = true
            }
        }

        private var dirty = true

        @Volatile
        private var uploading = false
        private var released = false

        /**
         * Unconditional captures after attach: composition lands on the recomposer's clock, so the first draws
         * may be empty with nothing marking them dirty afterwards.
         */
        private var warmup = 0

        private var scratch: Bitmap? = null
        private var pixels: ByteBuffer? = null

        /** The size [view] was last laid out at, so an unchanged one costs no layout pass. */
        private var laidOut: IntSize? = null

        /** Last capture time, to pace captures of a page the window already shows. */
        private var lastCaptureNanos = 0L

        /**
         * Consecutive capture failures at the current size, and passes with no content size; past the cap,
         * every-frame retries stop until the size changes.
         */
        private var failures = 0
        private var unsizedPasses = 0

        /** Last failure time; a page given up on is retried after a cool-down. */
        private var failedAtNanos = 0L

        private val gaveUp: Boolean
            get() = failures >= MAX_FAILURES || unsizedPasses >= MAX_UNSIZED_PASSES

        fun markDirty() {
            dirty = true
        }

        /** A capture that threw, on this thread or in its upload. Logged once per run of them. */
        fun failed(e: Throwable) {
            if (failures++ == 0) Log.w(TAG, "capture failed", e)
            failedAtNanos = System.nanoTime()
            dirty = true
            // Most likely out of memory: give the scratch back rather than hold it for a retry.
            dropScratch()
        }

        /**
         * Whether the pass must run every frame for this page. Always for a window-carried page: the handover
         * must land on the frame it starts moving, or a fast turn briefly stacks it over the scrolling page.
         * Otherwise only on-screen pages, plus any not yet through [warmup]; off-screen pages get one frame.
         */
        fun needsEveryFrame(): Boolean {
            if (released) return false
            if (placement != null) return true
            if (gaveUp) return false
            if (!page.hasSnapshot || warmup > 0) return true
            return page.isOnScreen
        }

        /** Draws and uploads the composition if anything changed since the last frame. */
        fun captureIfDirty() {
            if (released || page.destroyed) return
            // The renderer singleton frees every texture when its surface goes; capturing after that
            // segfaults the render thread rather than throwing.
            if (!WebGpuRenderer.isAvailable) return
            if (uploading) return

            // Before the size is resolved, because a page wrapping its content has nothing to
            // measure until the view is attached and the composition has landed in it.
            if (view.parent == null) {
                warmup = WARMUP_FRAMES
                addView(view)
                placeView()
            }

            // Every pass: rotation, dual-page switches and content changes all resize it.
            val footprint = resolveSize() ?: run {
                unsizedPasses++
                return
            }
            unsizedPasses = 0
            if (footprint.width != page.width || footprint.height != page.height) {
                page.resize(footprint)
                rehome(page)
                dirty = true
                // A new size is a new chance.
                failures = 0
            }
            if (failures >= MAX_FAILURES) {
                // Transient shortages (mostly memory) pass; retry at idle pace rather than leave the page blank.
                if (System.nanoTime() - failedAtNanos < FAILURE_COOLDOWN_NANOS) return
                failures = MAX_FAILURES - 1
                dirty = true
            }

            // Only on-screen pages capture every frame; off screen an endless progress ring would cost an
            // upload per frame, competing with decodes. Not before [warmup]: early captures are often blank.
            if (!page.isOnScreen && page.hasSnapshot && warmup == 0) {
                // Captured and hidden: drop the scratch rather than hold it for a page that may never show.
                if (placement == null) dropScratch()
                return
            }

            if (warmup > 0) {
                warmup--
                dirty = true
            }
            if (!dirty) return

            // While the window shows it live, the snapshot only serves a handover, so captures are paced.
            if (placement != null && warmup == 0 && page.hasSnapshot) {
                val now = System.nanoTime()
                if (now - lastCaptureNanos < PLACED_CAPTURE_INTERVAL_NANOS) return
            }

            // Compose lays out its content from the draw, but not the container, so size it here.
            layoutContent()

            // Cleared only once a frame is sent: a capture skipped for an in-flight upload must retry, or
            // the drag's settled frame never reaches the screen.
            dirty = false
            upload()
        }

        /**
         * [ImagePage.Compose.size] in this viewer's slot: the viewport, halved widthwise for a spread side.
         * Null while the viewport is unsized.
         */
        private fun resolveSize(): IntSize? {
            val state = page.parent ?: return null
            val continuous = state is ImageViewerContinuousState
            val maxWidth = if (isSpreadSide) state.width / 2 else state.width
            // Paged: below an avoided cutout, or a full-height page pans by its height ([ImagePage.nudgedYBounds]).
            val maxHeight = if (continuous) state.height else state.viewportHeight.toInt()
            if (maxWidth <= 0 || maxHeight <= 0) return null

            val resolved = when (val size = page.size) {
                is ComposePageSize.Of -> IntSize(size.width, size.height)
                // Continuous viewers fit pages to viewport width, so self-sized content would be scaled up;
                // it claims the slot instead. Paged viewers measure, so a card costs a card-sized capture.
                // A spread side takes its half's width ([ImagePage.ImageSpread] draws it as that half)
                // but only its content's height, so it can rest centred clear of an avoided cutout.
                ComposePageSize.Auto -> if (continuous) {
                    IntSize(maxWidth, maxHeight)
                } else {
                    measure(maxWidth, maxHeight, exactWidth = isSpreadSide)
                }
                // Width is the slot's for the same reason; only height is the content's.
                ComposePageSize.Wrap -> measure(maxWidth, maxHeight, exactWidth = continuous || isSpreadSide)
            } ?: return null
            // The snapshot is one texture ([Mipmap]'s drawable constructor) padded to whole copy units;
            // [Of] sizes beyond that or under a pixel are clamped, not failed.
            val limit = maxCaptureSide()
            return IntSize(resolved.width.coerceIn(1, limit), resolved.height.coerceIn(1, limit))
        }

        private fun maxCaptureSide(): Int =
            (min(WebGpuRenderer.maxTextureDimension2D, MAX_CAPTURE_SIDE) / 4 * 4).coerceAtLeast(4)

        /** The content's own size within [maxWidth] x [maxHeight], or null before it has one. */
        private fun measure(maxWidth: Int, maxHeight: Int, exactWidth: Boolean): IntSize? {
            view.measure(
                MeasureSpec.makeMeasureSpec(
                    maxWidth,
                    if (exactWidth) MeasureSpec.EXACTLY else MeasureSpec.AT_MOST,
                ),
                MeasureSpec.makeMeasureSpec(maxHeight, MeasureSpec.AT_MOST),
            )
            val width = view.measuredWidth
            val height = view.measuredHeight
            // Nothing composed yet. Taking a zero size would pin the page at a pixel and satisfy the
            // has-a-frame test, after which nothing re-measures. Retry next frame.
            if (width <= 0 || height <= 0) return null
            return IntSize(width.coerceAtMost(maxWidth), height.coerceAtMost(maxHeight))
        }

        /** Only on size change or request: setting layout params relayouts the whole window. */
        private fun layoutContent() {
            val size = IntSize(page.width, page.height)
            if (size == laidOut && !view.isLayoutRequested) return
            if (size != laidOut) {
                view.layoutParams = LayoutParams(size.width, size.height)
                laidOut = size
            }
            view.measure(
                MeasureSpec.makeMeasureSpec(size.width, MeasureSpec.EXACTLY),
                MeasureSpec.makeMeasureSpec(size.height, MeasureSpec.EXACTLY),
            )
            view.layout(0, 0, size.width, size.height)
        }

        /** Draws the content into the scratch bitmap and uploads it as the page's image. */
        private fun upload() {
            val bitmap = scratchBitmap()
            val buffer = pixels ?: return
            val size = IntSize(bitmap.width, bitmap.height)

            canvas.setBitmap(bitmap)
            try {
                canvas.drawColor(page.backgroundColor ?: 0, PorterDuff.Mode.SRC)
                view.draw(canvas)
            } finally {
                canvas.setBitmap(null)
            }

            buffer.rewind()
            bitmap.copyPixelsToBuffer(buffer)
            buffer.rewind()

            val first = !page.hasSnapshot
            lastCaptureNanos = System.nanoTime()
            uploading = true
            scope.launch {
                try {
                    page.capture(buffer, size)
                    failures = 0
                    // [ImagePage.invalidate] only redraws a page already on screen; a first frame isn't, so
                    // ask the viewer directly.
                    if (first) page.parent?.invalidate()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Throwable) {
                    // A failed upload is a frame to redraw, not a crash.
                    post { failed(e) }
                }
            }.invokeOnCompletion {
                // Not in a finally: a launch on a cancelled scope never runs its body, leaving the flag set.
                uploading = false
            }
        }

        /** Reused: a Canvas per frame is an allocation per frame. */
        private val canvas = Canvas()

        /** Frees the scratch, which the next capture reallocates. Never while an upload reads it. */
        private fun dropScratch() {
            if (uploading) return
            scratch?.recycle()
            scratch = null
            pixels = null
        }

        /** The page-sized bitmap this draws into, reallocated only when the page resizes. */
        private fun scratchBitmap(): Bitmap {
            val existing = scratch
            if (existing != null && !existing.isRecycled &&
                existing.width == page.width && existing.height == page.height
            ) {
                return existing
            }
            existing?.recycle()
            // Dropped first so a throw below can't leave [scratch] pointing at the recycled bitmap.
            scratch = null
            pixels = null
            // ARGB_8888 is RGBA byte order, matching RGBA8Unorm; writeTexture needs a direct native-order
            // buffer. Sides are capped by [maxCaptureSide], so this fits an Int.
            val bitmap = Bitmap.createBitmap(page.width, page.height, Bitmap.Config.ARGB_8888)
            pixels = ByteBuffer.allocateDirect(page.width * page.height * Int.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            return bitmap.also { scratch = it }
        }

        /**
         * Whether a gesture can map onto the composition: only while the window carries this page. Otherwise
         * it's moving, a spread side, or behind a non-invertible transition shader, and the gesture is the
         * viewer's.
         */
        fun acceptsInput(): Boolean {
            if (released || page.destroyed) return false
            return placement != null
        }

        /**
         * Puts [placement] on the view as its transform, so the window draws it there and accessibility
         * reports the content where it's drawn rather than at the origin. Touches are mapped by hand and
         * skip it. A page the window isn't carrying is left out of accessibility.
         */
        fun placeView() {
            val placement = placement
            placingView = true
            try {
                if (placement == null) {
                    view.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
                    return
                }
                view.importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_AUTO
                view.pivotX = 0f
                view.pivotY = 0f
                view.translationX = placement.left
                view.translationY = placement.top
                view.scaleX = placement.scale
                view.scaleY = placement.scale
            } finally {
                placingView = false
            }
        }

        /** Through [placement], which the window draws the content through too. */
        fun screenToPage(): Matrix? = placement?.screenToPage()

        /** Drops view and bitmap; not the page's texture, which outlives this and may still be drawn. */
        fun release() {
            if (released) return
            released = true
            if (owner === this) {
                owner = null
                gestureMap = null
            }
            // Detaching disposes the composition, so an out-of-reach page costs nothing in the viewer's cache.
            if (built && view.parent != null) removeView(view)
            scratch?.recycle()
            scratch = null
            pixels = null
        }
    }

    /** The activity behind this view's context, which is every owner a composition asks for. */
    private inline fun <reified T> Context.owner(): T? {
        var context: Context? = this
        while (context != null) {
            if (context is T) return context
            context = (context as? ContextWrapper)?.baseContext
        }
        return null
    }

    private companion object {
        const val TAG = "ComposePageHost"

        /** Idle pass interval: cheap for an idle reader, short enough to draw a page before it's turned to. */
        const val IDLE_INTERVAL_MS = 200L

        /** How often a page the window is showing refreshes the snapshot a handover lands on. */
        const val PLACED_CAPTURE_INTERVAL_NANOS = 50_000_000L

        /** About a third of a second at sixty frames. */
        const val REACH_BACKSTOP_PASSES = 20

        const val FAILURE_COOLDOWN_NANOS = 2_000_000_000L

        /** [Mipmap]'s drawable tile size: the snapshot is a single texture no larger. */
        const val MAX_CAPTURE_SIDE = 4096

        const val MAX_FAILURES = 3

        /** A second at sixty frames: long enough for any composition to land. */
        const val MAX_UNSIZED_PASSES = 60

        /** Frames captured unconditionally after an attach - see [PageRenderer.needsEveryFrame]. */
        const val WARMUP_FRAMES = 3

        /** Page offset below which a turn counts as settled, matching the viewer's own spring. */
        const val SETTLED = 0.002f

        /** Passes a continuous page has to hold its place before the window takes it. */
        const val STILL_PASSES = 3
    }
}

/** Where a carried page sits on the surface: its top-left, and screen pixels per page pixel. */
internal data class PagePlacement(val left: Float, val top: Float, val scale: Float) {

    /** This placement inverted, so a touch reaches the pixel of the composition it landed on. */
    fun screenToPage() = Matrix().apply {
        postTranslate(-left, -top)
        postScale(1f / scale, 1f / scale)
    }
}

/**
 * Where a page's footprint lands on the surface from its x, y and scale, as the paged viewer places it.
 *
 * Restates [ca.mpreg.webgpuviewer.renderer.Image.placement]: the window draws through it and touches
 * map back through it, so all three must agree. Null for an unsized viewport or page, or a collapsed
 * scale.
 */
internal fun pagePlacement(
    viewportWidth: Float,
    viewportHeight: Float,
    pageWidth: Int,
    pageHeight: Int,
    x: Float,
    y: Float,
    scale: Float,
    offsetX: Float,
    offsetY: Float,
): PagePlacement? {
    if (viewportWidth <= 0f || viewportHeight <= 0f || scale <= 0f) return null
    if (pageWidth <= 0 || pageHeight <= 0) return null

    return PagePlacement(
        left = viewportWidth * (0.5f + scale * ((x + offsetX) - 0.5f * pageWidth / viewportWidth)),
        top = viewportHeight * (0.5f + scale * ((y + offsetY) - 0.5f * pageHeight / viewportHeight)),
        scale = scale,
    )
}

/** [pagePlacement], inverted - where a touch on the surface lands in the page's own pixels. */
fun screenToPageMatrix(
    viewportWidth: Float,
    viewportHeight: Float,
    pageWidth: Int,
    pageHeight: Int,
    x: Float,
    y: Float,
    scale: Float,
    offsetX: Float,
    offsetY: Float,
): Matrix? = pagePlacement(
    viewportWidth, viewportHeight, pageWidth, pageHeight, x, y, scale, offsetX, offsetY,
)?.screenToPage()
