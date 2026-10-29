package ca.mpreg.webgpuviewer.renderer

import kotlin.math.truncate

/**
 * mpv user shader size expressions (`//!WHEN`, `//!WIDTH`, `//!HEIGHT`, and `//!HALO`):
 * space-separated RPN. Numbers, `NAME.w`/`NAME.h` for a texture's size, a bare `NAME` for a param,
 * `!`, binary `+ - * / % > < >= <= == != && ||`, and ternary `?`: `c a b ?` is `a` if `c` is
 * nonzero, else `b`. Comparisons give 1 or 0; `//!WHEN` runs the pass when nonzero.
 */
internal class SzExpr private constructor(val source: String, private val tokens: List<Token>) {

    private sealed class Token {
        class Number(val value: Double) : Token()
        class Variable(val name: String) : Token()
        class Operator(val name: String) : Token()
    }

    /**
     * With [integerDivision], `/` truncates, as sizes need: `w 3 + 4 / 4 *` rounds [w] up to a
     * multiple of 4. `//!WHEN` keeps the fraction, for `OUTPUT.w HOOKED.w /`.
     */
    fun eval(variables: Map<String, Double>, integerDivision: Boolean = false): Double {
        val stack = ArrayDeque<Double>()
        for (token in tokens) {
            when (token) {
                is Token.Number -> stack.addLast(token.value)
                is Token.Variable -> stack.addLast(variables[token.name] ?: Double.NaN)
                is Token.Operator -> when (token.name) {
                    "?" -> {
                        val b = stack.removeLast()
                        val a = stack.removeLast()
                        stack.addLast(if (truthy(stack.removeLast())) a else b)
                    }

                    "!" -> stack.addLast(if (truthy(stack.removeLast())) 0.0 else 1.0)
                    else -> {
                        val b = stack.removeLast()
                        val a = stack.removeLast()
                        stack.addLast(
                            if (integerDivision && token.name == "/") truncate(a / b)
                            else binary(token.name, a, b)
                        )
                    }
                }
            }
        }
        return stack.first()
    }

    companion object {
        private val BINARY =
            setOf("+", "-", "*", "/", "%", ">", "<", ">=", "<=", "==", "!=", "&&", "||")

        private val VARIABLE = Regex("""[A-Za-z_][A-Za-z0-9_]*(\.[wh])?""")

        /** Nonzero and a number, as mpv reads a condition. */
        private fun truthy(v: Double) = v != 0.0 && !v.isNaN()

        private fun bool(v: Boolean) = if (v) 1.0 else 0.0

        private fun binary(op: String, a: Double, b: Double): Double = when (op) {
            "+" -> a + b
            "-" -> a - b
            "*" -> a * b
            "/" -> a / b
            "%" -> a % b
            ">" -> bool(a > b)
            "<" -> bool(a < b)
            ">=" -> bool(a >= b)
            "<=" -> bool(a <= b)
            "==" -> bool(a == b)
            "!=" -> bool(a != b)
            "&&" -> bool(truthy(a) && truthy(b))
            else -> bool(truthy(a) || truthy(b))
        }

        /** Throws if [text] is not one expression, or names a variable [known] does not accept. */
        fun parse(text: String, known: (variable: String) -> Boolean): SzExpr {
            val tokens = ArrayList<Token>()
            var depth = 0
            for (word in text.trim().split(Regex("""\s+"""))) {
                require(word.isNotEmpty()) { "expression \"$text\": empty" }
                val pops = when {
                    word == "!" -> 1
                    word in BINARY -> 2
                    word == "?" -> 3
                    else -> 0
                }
                if (pops > 0) {
                    tokens.add(Token.Operator(word))
                } else if (VARIABLE.matches(word)) {
                    require(known(word)) { "expression \"$text\": unknown $word" }
                    tokens.add(Token.Variable(word))
                } else {
                    val value = word.toDoubleOrNull()?.takeIf { it.isFinite() }
                        ?: throw IllegalArgumentException("expression \"$text\": unexpected $word")
                    tokens.add(Token.Number(value))
                }
                require(depth >= pops) { "expression \"$text\": $word is missing operands" }
                depth += 1 - pops
            }
            require(depth == 1) { "expression \"$text\": must leave exactly one value" }
            return SzExpr(text, tokens)
        }
    }
}
