// A test harness small enough to need no dependency and no network fetch, which matters because
// the point of these tests is that they run anywhere a host compiler does - no NDK, no device,
// no GPU.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace check {

struct Case {
  const char *name;
  void (*run)();
};

inline std::vector<Case> &cases() {
  static std::vector<Case> all;
  return all;
}

inline int &failures() {
  static int count = 0;
  return count;
}

inline const char *&current() {
  static const char *name = "";
  return name;
}

struct Register {
  Register(const char *name, void (*run)()) { cases().push_back({name, run}); }
};

inline void fail(const char *file, int line, const std::string &what) {
  ++failures();
  std::fprintf(stderr, "  FAIL %s\n    %s:%d: %s\n", current(), file, line, what.c_str());
}

inline void isTrue(bool value, const char *expr, const char *file, int line) {
  if (!value) fail(file, line, std::string("expected true: ") + expr);
}

template <typename A, typename B>
void equal(const A &a, const B &b, const char *expr, const char *file, int line) {
  if (!(a == b)) {
    fail(file, line, std::string(expr) + " -> " + std::to_string(a) + " != " + std::to_string(b));
  }
}

inline void near(double a, double b, double tolerance, const char *expr, const char *file, int line) {
  if (std::fabs(a - b) > tolerance) {
    fail(file, line, std::string(expr) + " -> " + std::to_string(a) + " differs from " +
                         std::to_string(b) + " by more than " + std::to_string(tolerance));
  }
}

inline int main() {
  for (const Case &c : cases()) {
    current() = c.name;
    const int before = failures();
    c.run();
    if (failures() == before) std::printf("  ok   %s\n", c.name);
  }
  std::printf("%d case(s), %d failure(s)\n", static_cast<int>(cases().size()), failures());
  return failures() == 0 ? 0 : 1;
}

} // namespace check

#define TEST(name)                                                                                 \
  static void name();                                                                              \
  static check::Register reg_##name(#name, name);                                                  \
  static void name()

#define CHECK(expr) check::isTrue((expr), #expr, __FILE__, __LINE__)
#define CHECK_EQ(a, b) check::equal((a), (b), #a " == " #b, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) check::near((a), (b), (tol), #a " ~ " #b, __FILE__, __LINE__)
