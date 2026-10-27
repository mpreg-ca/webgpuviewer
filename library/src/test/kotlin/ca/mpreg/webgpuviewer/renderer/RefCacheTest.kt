package ca.mpreg.webgpuviewer.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class RefCacheTest {
    private class Value(val name: String)

    private val closed = ArrayList<String>()
    private val forgotten = ArrayList<String>()
    private var built = 0

    private fun cache(maxIdle: Int = 2) =
        RefCache<Value>(maxIdle, close = { closed += it.name }, forget = { forgotten += it.name })

    private fun RefCache<Value>.get(key: String) = acquire(key) { built++; Value(key) }

    @Test
    fun sharesWhileHeld() {
        val cache = cache()
        val a = cache.get("a")
        assertSame(a, cache.get("a"))
        assertEquals(1, built)
    }

    @Test
    fun keepsReleasedForReuse() {
        val cache = cache()
        val a = cache.get("a")
        cache.release(a)
        assertSame(a, cache.get("a"))
        assertEquals(1, built)
        assertTrue(closed.isEmpty())
    }

    @Test
    fun closesLeastRecentlyReleasedPastLimit() {
        val cache = cache(maxIdle = 2)
        val entries = listOf("a", "b", "c").map { cache.get(it) }
        entries.forEach { cache.release(it) }
        assertEquals(listOf("a"), closed)
        cache.get("a")
        assertEquals(4, built)
        cache.get("b")
        assertEquals(4, built)
    }

    @Test
    fun heldIsNeverClosed() {
        val cache = cache(maxIdle = 0)
        val a = cache.get("a")
        cache.get("a")
        cache.release(a)
        assertTrue(closed.isEmpty())
        cache.release(a)
        assertEquals(listOf("a"), closed)
    }

    @Test
    fun evictedHeldClosesAtLastReleaseAndIsRebuilt() {
        val cache = cache()
        val a = cache.get("a")
        cache.evict(a)
        val fresh = cache.get("a")
        assertEquals(2, built)
        cache.release(a)
        assertEquals(listOf("a"), closed)
        assertSame(fresh, cache.get("a"))
    }

    @Test
    fun evictedIdleClosesNow() {
        val cache = cache()
        val a = cache.get("a")
        cache.release(a)
        cache.evict(a)
        assertEquals(listOf("a"), closed)
        cache.get("a")
        assertEquals(2, built)
    }

    @Test
    fun evictingAStaleEntryLeavesTheNewOne() {
        val cache = cache()
        val a = cache.get("a")
        cache.evict(a)
        val fresh = cache.get("a")
        cache.evict(a)
        assertSame(fresh, cache.get("a"))
    }

    @Test
    fun clearForgetsInsteadOfClosing() {
        val cache = cache()
        val held = cache.get("a")
        cache.release(cache.get("b"))
        cache.clear()
        assertEquals(listOf("b"), forgotten)
        cache.release(held)
        assertEquals(listOf("b", "a"), forgotten)
        assertTrue(closed.isEmpty())
        cache.get("a")
        assertEquals(3, built)
    }

    @Test(expected = IllegalStateException::class)
    fun overReleaseThrows() {
        val cache = cache()
        val a = cache.get("a")
        cache.release(a)
        cache.release(a)
    }
}
