package ca.mpreg.webgpuviewer.renderer

import androidx.webgpu.TextureFormat
import kotlin.math.ceil
import kotlin.math.floor

/**
 * A parsed WGSL network file: passes with mpv user shader headers.
 *
 * Validated here, with messages naming the pass, so a bad file fails when its [ImageFilterWgsl] is
 * built, not as a GPU error on the first block.
 */
internal class WgslNetwork(
    val stages: List<Stage>,
    /** Input pixels of page around an output pixel it reads, for passes without `//!HALO`. */
    private val fileHalo: SzExpr?,
    val params: List<ImageFilterWgsl.Param>,
) {

    enum class Kind { Sampled, Storage, Sampler }

    class Binding(val index: Int, val kind: Kind)

    /** One pass: an mpv user shader header and its WGSL. */
    class Stage(
        val name: String,
        val code: String,
        /** Textures for the pass's `texture_2d` bindings, in binding order. */
        val binds: List<String>,
        /** What it writes, or null to replace `HOOKED`. */
        val save: String?,
        val width: SzExpr,
        val height: SzExpr,
        val `when`: SzExpr?,
        val halo: SzExpr?,
        val bindings: List<Binding>,
        val storageFormat: Int,
        val blockX: Int,
        val blockY: Int,
        /** Indices in [params] of those it declares as `override`s. */
        val params: List<Int>,
    )

    /** A texture as a pass sees it. The tile itself has no [format]. */
    class Ref(val width: Int, val height: Int, val format: Int?, val written: Int) {
        var slot = -1

        /** Last step to read it; [Int.MAX_VALUE] for the network's result. */
        var lastRead = written
    }

    /** One pass, sized for a tile. */
    class Step(
        val stage: Stage,
        val width: Int,
        val height: Int,
        val reads: List<Ref>,
        val writes: Ref
    )

    /** A texture shared by [Ref]s whose lifetimes don't overlap. */
    class SlotSpec(val width: Int, val height: Int, val format: Int)

    /** A tile's passes, the sizes they write and the textures holding them. */
    class Plan(
        val input: Ref,
        val steps: List<Step>,
        val slots: List<SlotSpec>,
        /** What the last pass to replace `HOOKED` wrote. */
        val result: Ref,
    ) {
        /** Fixes the textures and bindings: a plan that matches needs nothing rebuilt. */
        val signature: String
            get() = steps.joinToString(",") { "${it.stage.name}:${it.width}x${it.height}" }
    }

    /** The passes whose `//!WHEN` accepts a tile at page [scale]. */
    fun active(scale: Float): List<Stage> {
        val env = mapOf(
            "HOOKED.w" to NOMINAL_TILE / scale.toDouble(),
            "HOOKED.h" to NOMINAL_TILE / scale.toDouble(),
            "OUTPUT.w" to NOMINAL_TILE.toDouble(),
            "OUTPUT.h" to NOMINAL_TILE.toDouble(),
        )
        // NaN, from a zero size, also rejects.
        return stages.filter { s ->
            val `when` = s.`when`?.eval(env) ?: 1.0
            `when` != 0.0 && !`when`.isNaN()
        }
    }

    fun defaults() = FloatArray(params.size) { params[it].default }

    /**
     * Input pixels of page around the tile needed: the most any of [active] reads, with [params] at
     * [values]. Throws if a `//!HALO` comes to less than 0 or more than [MAX_HALO].
     */
    fun halo(active: List<Stage>, values: FloatArray): Int {
        val env = params.indices.associate { params[it].name to values[it].toDouble() }
        return active.maxOfOrNull { stage ->
            val expr = stage.halo ?: fileHalo ?: return@maxOfOrNull 0
            val halo = ceil(expr.eval(env))
            require(halo in 0.0..MAX_HALO.toDouble()) {
                "${stage.name}: //!HALO ${expr.source} is $halo, not 0 to $MAX_HALO"
            }
            halo.toInt()
        } ?: 0
    }

    /**
     * Size [active] for a [side]-pixel square tile at page [scale], reusing each texture once its last
     * reader has run. Throws if a pass reads something unwritten, or none replaces `HOOKED`.
     */
    fun plan(active: List<Stage>, side: Int, scale: Float): Plan {
        val input = Ref(side, side, null, -1)
        val names = LinkedHashMap<String, Ref>().apply { put("HOOKED", input) }
        val steps = ArrayList<Step>()
        active.forEachIndexed { i, stage ->
            val env = HashMap<String, Double>()
            env["OUTPUT.w"] = side * scale.toDouble()
            env["OUTPUT.h"] = side * scale.toDouble()
            for ((name, ref) in names) {
                env["$name.w"] = ref.width.toDouble()
                env["$name.h"] = ref.height.toDouble()
            }
            val reads = stage.binds.map { b ->
                names[b] ?: bad("${stage.name}: //!BIND $b was not written before it")
            }
            reads.forEach { it.lastRead = i }
            val w = floor(stage.width.eval(env, true))
            val h = floor(stage.height.eval(env, true))
            require(w.isFinite() && h.isFinite() && w >= 1 && h >= 1 && w <= Int.MAX_VALUE && h <= Int.MAX_VALUE) {
                "${stage.name}: ${w}x$h is not a size"
            }
            val writes = Ref(w.toInt(), h.toInt(), stage.storageFormat, i)
            names[stage.save ?: "HOOKED"] = writes
            steps.add(Step(stage, w.toInt(), h.toInt(), reads, writes))
        }
        val result = names.getValue("HOOKED")
        require(result !== input) { "no pass without //!SAVE ran, so nothing replaced HOOKED" }
        result.lastRead = Int.MAX_VALUE

        val slots = ArrayList<SlotSpec>()
        val resident = ArrayList<Ref>()
        steps.forEachIndexed { i, step ->
            val out = step.writes
            val free = slots.indices.firstOrNull { k ->
                val slot = slots[k]
                slot.width == out.width && slot.height == out.height && slot.format == out.format &&
                        resident[k].lastRead < i
            } ?: -1
            if (free >= 0) {
                resident[free] = out
                out.slot = free
            } else {
                out.slot = slots.size
                slots.add(SlotSpec(out.width, out.height, out.format!!))
                resident.add(out)
            }
        }
        return Plan(input, steps, slots, result)
    }

    /** [format] and [access]: the type's template arguments, else empty. */
    private class Found(val binding: Binding, val format: String, val access: String)

    private class Draft {
        var desc: String? = null
        val hooks = ArrayList<String>()
        val binds = ArrayList<String>()
        var save: String? = null
        var width: String? = null
        var height: String? = null
        var `when`: String? = null
        var components: Int? = null
        var compute: List<Int>? = null
        var halo: String? = null
        val code = StringBuilder()

        /** Set by any non-blank body line; the next directive then starts another pass. */
        var hasBody = false
    }

    private class ParamDraft(val name: String) {
        var desc: String? = null
        var type: String? = null
        var default: String? = null
        var minimum: String? = null
        var maximum: String? = null
    }

    companion object {
        private const val MAX_HALO = 256
        private const val MAX_PASSES = 64

        /** Bounds a bad file. */
        private const val MAX_PARAMS = 64

        /** Compatibility mode's `maxComputeInvocationsPerWorkgroup`, the lowest a device has. */
        private const val MAX_INVOCATIONS = 128

        /**
         * The tile `//!WHEN` is evaluated against. Only ratios mean anything: `OUTPUT.w HOOKED.w /`
         * is the page scale; an absolute size is this one's.
         */
        private const val NOMINAL_TILE = 256

        /** Hooks a pass may name; both mean the tile between draw and resize. */
        private val HOOKS = setOf("MAIN", "RGB")

        fun bytesPerTexel(format: Int): Int = when (format) {
            TextureFormat.RGBA8Unorm -> 4
            TextureFormat.RGBA16Float -> 8
            else -> 16
        }

        private val NAME = Regex("""[A-Za-z_][A-Za-z0-9_]*""")

        private val DIRECTIVE = Regex("""^\s*//!(\w+)\s*(.*?)\s*$""")
        private val BINDING = Regex(
            """@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]*>)?\s+\w+\s*:\s*""" +
                    """(texture_storage_2d|texture_2d|sampler)\b(?:<\s*(\w+)(?:\s*,\s*(\w+))?)?"""
        )
        private val GROUP = Regex("""@group\(""")
        private val OVERRIDE = Regex("""\boverride\s+(\w+)""")
        private val COMPUTE = Regex("""@compute[^{]*?\bfn\s+main\s*\(""")
        private val WORKGROUP = Regex("""@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+))?""")
        private val INTEGER = Regex("""[+-]?\d+""")

        private val FORMATS = mapOf(
            "rgba16float" to TextureFormat.RGBA16Float,
            "rgba8unorm" to TextureFormat.RGBA8Unorm,
        )

        /** Per-pass directives; one after a pass's body starts the next pass. */
        private val PASS_KEYS =
            setOf(
                "DESC",
                "HOOK",
                "BIND",
                "SAVE",
                "WIDTH",
                "HEIGHT",
                "WHEN",
                "COMPONENTS",
                "COMPUTE"
            )

        private fun toIntOrNull(text: String): Int? =
            if (INTEGER.matches(text)) text.toIntOrNull() else null

        private fun param(draft: ParamDraft): ImageFilterWgsl.Param {
            val name = draft.name
            val type = when (draft.type ?: "float") {
                "float" -> ImageFilterWgsl.ParamType.Float
                "int" -> ImageFilterWgsl.ParamType.Int
                "uint" -> ImageFilterWgsl.ParamType.UInt
                else -> bad("PARAM $name: TYPE ${draft.type}: float, int or uint")
            }
            fun number(text: String): Float = when (type) {
                ImageFilterWgsl.ParamType.Float -> text.toFloatOrNull()?.takeIf { it.isFinite() }
                ImageFilterWgsl.ParamType.Int -> toIntOrNull(text)?.toFloat()
                ImageFilterWgsl.ParamType.UInt -> toIntOrNull(text)?.takeIf { it >= 0 }?.toFloat()
            } ?: bad("PARAM $name: $text is not a ${type.wgsl}")
            val min = draft.minimum?.let(::number) ?: if (type == ImageFilterWgsl.ParamType.UInt) 0f else null
            val max = draft.maximum?.let(::number)
            val default = draft.default ?: bad("PARAM $name: no //!DEFAULT")
            val value = number(default)
            require((min == null || value >= min) && (max == null || value <= max)) {
                "PARAM $name: $default is not within ${min ?: "-inf"} to ${max ?: "inf"}"
            }
            return ImageFilterWgsl.Param(name, draft.desc.orEmpty(), type, min, max, value)
        }

        fun parse(source: String): WgslNetwork {
            var fileHalo: String? = null
            val drafts = ArrayList<Draft>()
            var current: Draft? = null
            fun next(): Draft {
                require(drafts.size < MAX_PASSES) { "more than $MAX_PASSES passes" }
                return Draft().also { drafts.add(it); current = it }
            }
            val paramDrafts = LinkedHashMap<String, ParamDraft>()
            var param: ParamDraft? = null
            for (line in source.split(Regex("""\r\n|\r|\n"""))) {
                val match = DIRECTIVE.matchEntire(line)
                val open = param
                if (open != null) {
                    // Its block ends at the first line not its own, so a pass's //!DESC right
                    // after it would be taken as the param's.
                    val key = match?.groupValues?.get(1)
                    val value = match?.groupValues?.get(2)
                    when {
                        key == "DESC" && open.desc == null -> open.desc = value
                        key == "TYPE" && open.type == null -> open.type = value
                        key == "DEFAULT" && open.default == null -> open.default = value
                        key == "MINIMUM" && open.minimum == null -> open.minimum = value
                        key == "MAXIMUM" && open.maximum == null -> open.maximum = value
                        else -> param = null
                    }
                    if (param != null) {
                        // Kept, as comments, so the device's line numbers match the file's.
                        current?.code?.append(line)?.append('\n')
                        continue
                    }
                }
                if (match == null) {
                    current?.let {
                        it.code.append(line).append('\n')
                        if (line.isNotBlank()) it.hasBody = true
                    }
                    continue
                }
                val (key, value) = match.destructured

                if (key == "PARAM") {
                    require(NAME.matches(value)) { "PARAM $value: not a name" }
                    require(value !in paramDrafts) { "PARAM $value: declared twice" }
                    require(paramDrafts.size < MAX_PARAMS) { "more than $MAX_PARAMS params" }
                    param = ParamDraft(value).also { paramDrafts[value] = it }
                    current?.code?.append(line)?.append('\n')
                    continue
                }

                if (key == "HALO") {
                    // Parsed once every param is known: one may come later in the file.
                    val halo = value
                    // Before any pass: every pass's default. In a header: that pass's.
                    val draft = current
                    if (draft == null || draft.hasBody) {
                        if (drafts.isEmpty()) {
                            fileHalo = halo
                            continue
                        }
                        next().halo = halo
                    } else {
                        draft.halo = halo
                    }
                    continue
                }

                require(key in PASS_KEYS) { "unknown //!$key" }
                val draft = current?.takeIf { !it.hasBody } ?: next()
                when (key) {
                    "DESC" -> draft.desc = value
                    "HOOK" -> draft.hooks.add(value)
                    "BIND" -> {
                        require(NAME.matches(value)) { "BIND $value: not a name" }
                        draft.binds.add(value)
                    }

                    "SAVE" -> {
                        require(NAME.matches(value)) { "SAVE $value: not a name" }
                        draft.save = value
                    }

                    "WIDTH" -> draft.width = value
                    "HEIGHT" -> draft.height = value
                    "WHEN" -> draft.`when` = value
                    "COMPONENTS" -> draft.components = toIntOrNull(value)?.takeIf { it in 1..4 }
                        ?: bad("COMPONENTS $value: 1 to 4")

                    "COMPUTE" -> {
                        val numbers = value.split(Regex("""\s+""")).map { toIntOrNull(it) }
                        require((numbers.size == 2 || numbers.size == 4) && numbers.all { it != null && it >= 1 }) {
                            "COMPUTE $value: block width and height, then optionally threads"
                        }
                        draft.compute = numbers.map { it!! }
                    }
                }
            }
            require(drafts.isNotEmpty()) { "no pass" }

            // WebGPU rejects a constant its module doesn't declare.
            val names = paramDrafts.keys.toList()
            val declared = drafts.map { d ->
                OVERRIDE.findAll(d.code).map { names.indexOf(it.groupValues[1]) }.filter { it >= 0 }.distinct().toList()
            }
            val params = paramDrafts.values.map(::param)

            fun halo(text: String) = SzExpr.parse(text) { it in paramDrafts }

            val saved = HashSet<String>()
            val stages = drafts.mapIndexed { i, d ->
                val name = d.desc ?: "pass$i"
                val code = d.code.toString()
                require(d.hooks.all { it in HOOKS }) { "$name: //!HOOK must be MAIN or RGB" }
                val found = BINDING.findAll(code).map { m ->
                    val kind = when (m.groupValues[2]) {
                        "texture_2d" -> Kind.Sampled
                        "texture_storage_2d" -> Kind.Storage
                        else -> Kind.Sampler
                    }
                    Found(
                        Binding(toIntOrNull(m.groupValues[1]) ?: bad("$name: binding index"), kind),
                        m.groupValues[3], m.groupValues[4]
                    )
                }.toList()
                // Only group 0 is bound; any other would be missing at draw time.
                require(GROUP.findAll(code).count() == found.size) {
                    "$name: every binding must be `@group(0) @binding(n) var name: type;`"
                }
                require(found.map { it.binding.index }.toSet().size == found.size) {
                    "$name: two bindings share an index"
                }
                require(found.none { it.binding.kind == Kind.Sampled && it.format != "f32" }) {
                    "$name: texture_2d bindings must be texture_2d<f32>"
                }
                require(COMPUTE.containsMatchIn(code)) { "$name: no @compute fn main" }
                val storage = found.filter { it.binding.kind == Kind.Storage }
                require(storage.size == 1) { "$name: needs exactly one texture_storage_2d" }
                require(storage[0].access == "write") { "$name: the texture_storage_2d must be write-only" }
                require(found.count { it.binding.kind == Kind.Sampled } == d.binds.size) {
                    "$name: //!BIND count differs from its texture_2d bindings"
                }
                for (b in d.binds) require(b == "HOOKED" || b in saved) {
                    "$name: //!BIND $b is not HOOKED or an earlier //!SAVE"
                }
                val format =
                    FORMATS[storage[0].format] ?: bad("$name: storage format ${storage[0].format}")

                val workgroup = WORKGROUP.find(code) ?: bad("$name: no @workgroup_size")
                val wx = toIntOrNull(workgroup.groupValues[1]) ?: 0
                val wy = toIntOrNull(workgroup.groupValues[2].ifEmpty { "1" }) ?: 0
                require(wx >= 1 && wy >= 1) { "$name: @workgroup_size must be numbers" }
                require(wx * wy <= MAX_INVOCATIONS) { "$name: more than $MAX_INVOCATIONS invocations" }
                // //!COMPUTE's threads are ignored: @workgroup_size sets them.
                val blockX = d.compute?.get(0) ?: wx
                val blockY = d.compute?.get(1) ?: wy

                val sizes = { variable: String ->
                    val texture = variable.substringBefore('.')
                    texture == "HOOKED" || texture == "OUTPUT" || texture in saved
                }
                val nominal =
                    { variable: String -> variable.startsWith("HOOKED.") || variable.startsWith("OUTPUT.") }
                val save = d.save.takeIf { it != "HOOKED" }
                require(save != "OUTPUT") { "$name: //!SAVE OUTPUT is the target's size, not a texture" }
                val stage = Stage(
                    name, code, d.binds, save,
                    SzExpr.parse(d.width ?: "HOOKED.w", sizes),
                    SzExpr.parse(d.height ?: "HOOKED.h", sizes),
                    d.`when`?.let { SzExpr.parse(it, nominal) },
                    d.halo?.let(::halo),
                    found.map { it.binding }.sortedBy { it.index },
                    format, blockX, blockY, declared[i],
                )
                if (save != null) saved.add(save)
                stage
            }
            require(stages.any { it.save == null }) { "no pass is without //!SAVE, so nothing replaces HOOKED" }
            return WgslNetwork(stages, fileHalo?.let(::halo), params).also {
                // Catches a constant out of range; others are checked as values change.
                it.halo(stages, it.defaults())
            }
        }

        private fun bad(message: String): Nothing = throw IllegalArgumentException(message)
    }
}
