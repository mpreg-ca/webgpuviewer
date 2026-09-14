#!/usr/bin/env bash
# Builds and runs the host-side tests for the library's native code.
#
# These need no NDK, no device and no GPU: the pixel maths in src/main/cpp is plain C++, so it is
# compiled for the host and exercised directly. jni.h comes from the host JDK because the
# implementation files include it, not because anything here calls into a JVM.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
src="$here/library/src/main/cpp"
tests="$here/library/src/test/cpp"
out=${OUT_DIR:-$here/build/native-test}

jdk=${JAVA_HOME:-}
if [ -z "$jdk" ]; then
    for candidate in /usr/lib/jvm/default /usr/lib/jvm/java-*-openjdk; do
        [ -f "$candidate/include/jni.h" ] && jdk=$candidate && break
    done
fi
[ -n "$jdk" ] && [ -f "$jdk/include/jni.h" ] || {
    echo "no jni.h found; set JAVA_HOME to a JDK" >&2
    exit 1
}

mkdir -p "$out"
status=0
shopt -s nullglob
for test_file in "$tests"/*_test.cpp; do
    name=$(basename "$test_file" .cpp)
    echo "== $name"
    "${CXX:-g++}" -std=c++17 -Wall -Wextra -Wno-unused-parameter -O1 -g \
        -I"$jdk/include" -I"$jdk/include/linux" -I"$src" -I"$tests" \
        "$test_file" -o "$out/$name" -lpthread
    "$out/$name" || status=1
done

exit $status
