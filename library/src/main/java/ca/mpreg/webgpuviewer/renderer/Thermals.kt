package ca.mpreg.webgpuviewer.renderer

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import android.view.View
import androidx.annotation.RequiresApi

/**
 * How hard the tile worker may push a device that is heating up.
 *
 * A staged rescaler is sustained GPU work, not a one-off: [UpscalerArtCnn] runs nine compute
 * dispatches per tile, and a reader holds that for as long as the reading lasts. Left alone it
 * competes with the frame being presented right up until the platform throttles the whole GPU,
 * which costs far more than pacing would have.
 *
 * So the worker waits a little between tiles once the platform says the device is warm. The delay
 * is deliberately crude - the point is to stop adding heat, not to hit a target temperature - and
 * it is zero at [PowerManager.THERMAL_STATUS_NONE] and [PowerManager.THERMAL_STATUS_LIGHT], where
 * nothing is being throttled yet and slowing down would only make tiles arrive later for nothing.
 *
 * Attached like [Hdr]'s colour-mode host: a view registers on attach and clears on detach, so the
 * library never holds a [Context] of its own. Below API 29 there is no status to read and this
 * stays at zero.
 */
object Thermals {

    private const val TAG = "Thermals"

    /**
     * Milliseconds to wait after a staged tile, by [PowerManager] thermal status. MODERATE is the
     * first level at which the platform is actually throttling something, so it is the first that
     * gets a pause; by SEVERE the GPU is already being cut back and a longer pause costs little
     * that throttling would not have taken anyway.
     */
    private const val PAUSE_MODERATE_MS = 8L
    private const val PAUSE_SEVERE_MS = 24L
    private const val PAUSE_CRITICAL_MS = 60L

    private var manager: PowerManager? = null

    @Volatile
    private var status: Int = 0

    private val listener = PowerManager.OnThermalStatusChangedListener { value ->
        status = value
    }

    /**
     * Register [view] as the thermal host, or pass null to clear it. Safe to call repeatedly: a
     * second attach replaces the first, and the listener follows.
     */
    fun attachHost(view: View?) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        detachListener()
        if (view == null) return
        val pm = view.context.applicationContext
            .getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return
        manager = pm
        try {
            status = pm.currentThermalStatus
            pm.addThermalStatusListener(listener)
        } catch (e: Exception) {
            // A device that refuses the listener still reads currentThermalStatus above; losing
            // the updates only means the pause stays at whatever was read on attach.
            Log.w(TAG, "thermal status listener unavailable", e)
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun detachListener() {
        val pm = manager ?: return
        try {
            pm.removeThermalStatusListener(listener)
        } catch (e: Exception) {
            Log.w(TAG, "could not remove thermal status listener", e)
        }
        manager = null
    }

    /**
     * How long the worker should wait after finishing a staged tile. Zero whenever the device is
     * not being throttled, or on a platform that cannot say.
     */
    val stagedTilePauseMs: Long
        get() = when {
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q -> 0L
            status >= PowerManager.THERMAL_STATUS_CRITICAL -> PAUSE_CRITICAL_MS
            status >= PowerManager.THERMAL_STATUS_SEVERE -> PAUSE_SEVERE_MS
            status >= PowerManager.THERMAL_STATUS_MODERATE -> PAUSE_MODERATE_MS
            else -> 0L
        }
}
