package ca.mpreg.webgpuviewer.renderer

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class WgslNetworkTest {
    private fun res(n: String) = WgslNetwork::class.java.getResourceAsStream(n)!!.bufferedReader().readText()

    private fun fails(msg: String, src: String) {
        try {
            WgslNetwork.parse(src); fail("accepted: $msg")
        } catch (e: IllegalArgumentException) {
        }
    }

    private val known = { _: String -> true }

    private fun eval(text: String, vararg vars: Pair<String, Double>) = SzExpr.parse(text, known).eval(mapOf(*vars))

    private val pass = """
//!DESC a
//!HOOK MAIN
//!BIND HOOKED
//!COMPUTE 8 8
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var o: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {}
"""

    @Test
    fun expressions() {
        assertEquals(1.0, eval("OUTPUT.w HOOKED.w / 1.3 >", "OUTPUT.w" to 300.0, "HOOKED.w" to 200.0), 0.0)
        assertEquals(0.0, eval("OUTPUT.w HOOKED.w / 1.3 >", "OUTPUT.w" to 200.0, "HOOKED.w" to 200.0), 0.0)
        assertEquals(2.0, eval("1 2 3 ?"), 0.0)
        assertEquals(3.0, eval("0 2 3 ?"), 0.0)
        assertEquals(42.0, eval("HOOKED.w 2 *", "HOOKED.w" to 21.0), 0.0)
        assertEquals(1.0, eval("1 0 ||  0 !  &&"), 0.0)
        assertEquals(8.0, SzExpr.parse("HOOKED.w 3 + 4 / 4 *", known).eval(mapOf("HOOKED.w" to 7.0), true), 0.0)
        for (bad in listOf("", "1 +", "1 2", "1foo", "HOOKED.x", "1 2 3")) {
            try {
                SzExpr.parse(bad, known); fail("accepted \"$bad\"")
            } catch (e: IllegalArgumentException) {
            }
        }
    }

    @Test
    fun shipped() {
        val net = WgslNetwork.parse(res("identity.wgsl"))
        assertEquals(1, net.stages.size)
        assertEquals(1, net.active(0.5f).size)
        assertEquals(1, net.active(2f).size)
        assertEquals(0, net.halo(net.active(2f), net.defaults()))
        val plan = net.plan(net.active(2f), 100, 2f)
        assertEquals(100, plan.result.width)
        assertEquals(100, plan.result.height)
        assertEquals(1, plan.steps.size)
        assertTrue(plan.steps[0].reads.single() === plan.input)
        assertTrue(plan.steps[0].writes === plan.result)
        assertEquals(1, plan.slots.size)
    }

    @Test
    fun samplers() {
        for (name in listOf("catmull_rom.wgsl", "box_downscale.wgsl")) {
            val code = res(name)
            assertTrue(name, "fn resolve(src_start: vec2<f32>, scale: vec2<f32>)" in code && "//!" !in code)
            fails("a sampler is not a network", code)
        }
    }

    @Test
    fun optional() {
        val bare = WgslNetwork.parse(pass.replace("//!HOOK MAIN\n", "").replace("//!COMPUTE 8 8\n", "")).stages.single()
        assertEquals(8, bare.blockX)
        assertEquals(8, bare.blockY)
        // Block and threads differ: 16x16 output pixels per 8x8 workgroup.
        val blocks = WgslNetwork.parse(pass.replace("COMPUTE 8 8", "COMPUTE 16 16")).stages.single()
        assertEquals(16, blocks.blockX)

        // A later pass reads the second x.
        val first = pass.replace("//!COMPUTE", "//!SAVE x\n//!COMPUTE")
        val net = WgslNetwork.parse(first + first.replace("//!DESC a", "//!DESC b") + pass.replace("BIND HOOKED", "BIND x"))
        val plan = net.plan(net.stages, 10, 1f)
        assertTrue(plan.steps[2].reads.single() === plan.steps[1].writes)
    }

    @Test
    fun rejects() {
        WgslNetwork.parse(pass)
        fails("empty", "")
        fails("hook", pass.replace("//!HOOK MAIN", "//!HOOK LUMA"))
        fails("no workgroup size", pass.replace("@workgroup_size(8, 8)", ""))
        fails("too many invocations", pass.replace("COMPUTE 8 8", "COMPUTE 16 16").replace("(8, 8)", "(16, 16)"))
        fails("bind unknown", pass.replace("BIND HOOKED", "BIND nope"))
        fails("bind count", pass.replace("//!BIND HOOKED\n", ""))
        fails("all saved", pass.replace("//!COMPUTE", "//!SAVE x\n//!COMPUTE"))
        fails("save output", pass.replace("//!COMPUTE", "//!SAVE OUTPUT\n//!COMPUTE"))
        fails("group1", pass.replace("@group(0) @binding(0)", "@group(1) @binding(0)"))
        fails("dup idx", pass.replace("@binding(1)", "@binding(0)"))
        fails("read_write", pass.replace(", write>", ", read_write>"))
        fails("format", pass.replace("rgba16float", "r32float"))
        fails("u32", pass.replace("texture_2d<f32>", "texture_2d<u32>"))
        fails("no compute fn", pass.replace("@compute", ""))
        fails("halo", "//!HALO 99999\n$pass")
        fails("halo neg", "//!HALO -1\n$pass")
        fails("unknown directive", pass.replace("//!COMPUTE 8 8", "//!FOO 2\n//!COMPUTE 8 8"))
        fails("bad width", pass.replace("//!COMPUTE", "//!WIDTH HOOKED.w +\n//!COMPUTE"))
    }

    private val param = "//!PARAM g\n//!DESC grain\n//!DEFAULT 4.5\n//!MINIMUM 0\n//!MAXIMUM 10\n\n"

    private fun declares(vararg overrides: String) =
        pass.replace("@compute", overrides.joinToString("") { "$it\n" } + "@compute")

    private val withG = param + declares("override g: f32 = 1.0;")

    @Test
    fun params() {
        val n = "//!PARAM n\n//!TYPE uint\n//!DEFAULT 3\n"
        val net = WgslNetwork.parse(param + n + declares("override  n :u32;", "override g: f32 = 1.0;", "override own: f32 = 2.0;"))
        assertEquals(listOf("g", "n"), net.params.map { it.name })
        assertEquals("grain", net.params[0].description)
        assertEquals(4.5f, net.params[0].default)
        assertEquals(10f, net.params[0].maximum)
        assertEquals(ImageFilterWgsl.ParamType.UInt, net.params[1].type)
        assertEquals("", net.params[1].description)
        assertEquals(0f, net.params[1].minimum)
        assertEquals(null, net.params[1].maximum)
        assertEquals(3f, net.params[1].default)
        assertEquals(listOf(1, 0), net.stages.single().params)

        // Between passes too, without joining either; each pass gets only what it declares.
        val two = pass + withG.replace("//!DESC a", "//!DESC b")
        assertEquals(listOf(emptyList(), listOf(0)), WgslNetwork.parse(two).stages.map { it.params })
        // Undeclared: only the halo might read it.
        assertEquals(ImageFilterWgsl.ParamType.Float, WgslNetwork.parse(param + pass).params.single().type)
        // Untyped in WGSL.
        assertEquals(listOf(0), WgslNetwork.parse(param + declares("override g = 1.0;")).stages.single().params)

        fails("no default", param.replace("//!DEFAULT 4.5\n", "") + pass)
        fails("out of range", param.replace("4.5", "11") + pass)
        fails("int", param.replace("//!DEFAULT", "//!TYPE int\n//!DEFAULT") + pass)
        fails("type", param.replace("//!DEFAULT", "//!TYPE vec2\n//!DEFAULT") + pass)
        fails("twice", param + withG)
        fails("name", param.replace("PARAM g", "PARAM 1g") + pass)
        // Beside its override, and kept there so line numbers hold.
        val beside = pass.replace("@compute", param.trimEnd() + "\noverride g: f32;\n@compute")
        val inPass = WgslNetwork.parse(beside)
        assertEquals("grain", inPass.params.single().description)
        assertEquals(listOf(0), inPass.stages.single().params)
        assertTrue(inPass.stages.single().code, inPass.stages.single().code.startsWith(beside.substringAfter("//!COMPUTE 8 8\n")))
        WgslNetwork.parse(pass.replace("//!COMPUTE", "$param//!COMPUTE"))
        fails("too many", (0..64).joinToString("") { "//!PARAM p$it\n//!DEFAULT 0\n" } + pass)
    }

    @Test
    fun coerce() {
        val net = WgslNetwork.parse(
            param + "//!PARAM i\n//!TYPE int\n//!DEFAULT 2\n\n//!PARAM u\n//!TYPE uint\n//!DEFAULT 2\n\n" + pass
        )
        val (g, i, u) = net.params
        assertEquals(10f, g.coerce(20f))
        assertEquals(0f, g.coerce(-1f))
        assertEquals(0.25f, g.coerce(0.25f))
        assertEquals(3f, i.coerce(2.5f))
        assertEquals(-3f, i.coerce(-2.6f))
        // Unbounded, yet still what the WGSL type holds: past it, the pipeline fails.
        assertEquals(2147483520f, i.coerce(Float.POSITIVE_INFINITY))
        assertEquals(-2147483648f, i.coerce(-1e30f))
        assertEquals(0f, u.coerce(-5f))
        assertEquals(4294967040f, u.coerce(1e30f))
        val f = WgslNetwork.parse("//!PARAM f\n//!DEFAULT 0\n\n$pass").params.single()
        assertEquals(Float.MAX_VALUE, f.coerce(Float.POSITIVE_INFINITY))
    }

    @Test
    fun haloExpression() {
        // Before the param it names.
        val net = WgslNetwork.parse("//!HALO g 3 * 1 +\n$withG")
        assertEquals(15, net.halo(net.stages, net.defaults()))
        assertEquals(31, net.halo(net.stages, floatArrayOf(10f)))
        try {
            net.halo(net.stages, floatArrayOf(-1f)); fail("accepted a negative halo")
        } catch (e: IllegalArgumentException) {
        }
        fails("unknown param", "//!HALO h\n$withG")
        fails("texture size", "//!HALO HOOKED.w\n$pass")
        fails("default out of range", "//!HALO g 100 *\n$withG")
    }
}
