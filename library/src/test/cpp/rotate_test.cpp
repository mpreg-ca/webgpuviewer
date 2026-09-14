// Tests for the quarter-turn rotation.
//
// A transpose is the kind of code that looks right and is off by one along an edge, or is correct
// for a square and wrong for everything else, so the cases below are deliberately non-square. The
// implementation is included rather than linked for the reason given in trim_test.cpp.

#include "rotate.cpp" // NOLINT(bugprone-suspicious-include)

#include "check.h"

#include <vector>

namespace {

/** A `width` x `height` image whose every pixel says where it came from. */
std::vector<uint8_t> ramp(int width, int height) {
  std::vector<uint8_t> out(static_cast<size_t>(width) * height * 4);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t *p = out.data() + (static_cast<size_t>(y) * width + x) * 4;
      p[0] = static_cast<uint8_t>(x);
      p[1] = static_cast<uint8_t>(y);
      p[2] = 0;
      p[3] = 255;
    }
  }
  return out;
}

/** The pixel at (x, y) of a `width`-wide RGBA image. */
const uint8_t *at(const std::vector<uint8_t> &img, int width, int x, int y) {
  return img.data() + (static_cast<size_t>(y) * width + x) * 4;
}

} // namespace

TEST(clockwiseSendsTheFirstColumnToTheLastRow) {
  const int w = 5, h = 3;
  auto src = ramp(w, h);
  std::vector<uint8_t> dst(src.size());
  rotate_quarter(src.data(), dst.data(), w, h, 4, true);

  // Destination is h x w. Source (x, y) lands at (h - 1 - y, x).
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t *got = at(dst, h, h - 1 - y, x);
      CHECK_EQ(static_cast<int>(got[0]), x);
      CHECK_EQ(static_cast<int>(got[1]), y);
    }
  }
}

TEST(anticlockwiseIsTheMirrorOfClockwise) {
  const int w = 5, h = 3;
  auto src = ramp(w, h);
  std::vector<uint8_t> dst(src.size());
  rotate_quarter(src.data(), dst.data(), w, h, 4, false);

  // Source (x, y) lands at (y, w - 1 - x).
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t *got = at(dst, h, y, w - 1 - x);
      CHECK_EQ(static_cast<int>(got[0]), x);
      CHECK_EQ(static_cast<int>(got[1]), y);
    }
  }
}

TEST(fourTurnsComeBackToTheStart) {
  const int w = 7, h = 4;
  auto a = ramp(w, h);
  std::vector<uint8_t> b(a.size()), c(a.size()), d(a.size()), e(a.size());
  rotate_quarter(a.data(), b.data(), w, h, 4, true);
  rotate_quarter(b.data(), c.data(), h, w, 4, true);
  rotate_quarter(c.data(), d.data(), w, h, 4, true);
  rotate_quarter(d.data(), e.data(), h, w, 4, true);
  CHECK(e == a);
}

TEST(oppositeTurnsUndoEachOther) {
  const int w = 6, h = 9;
  auto a = ramp(w, h);
  std::vector<uint8_t> b(a.size()), c(a.size());
  rotate_quarter(a.data(), b.data(), w, h, 4, true);
  rotate_quarter(b.data(), c.data(), h, w, 4, false);
  CHECK(c == a);
}

TEST(aTileBoundaryIsNotASeam) {
  // Deliberately either side of the 32-pixel tiling, where an off-by-one would hide.
  for (int w : {31, 32, 33, 64, 65}) {
    for (int h : {1, 31, 32, 33}) {
      auto src = ramp(w, h);
      std::vector<uint8_t> dst(src.size());
      rotate_quarter(src.data(), dst.data(), w, h, 4, true);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          const uint8_t *got = at(dst, h, h - 1 - y, x);
          CHECK_EQ(static_cast<int>(got[0]), static_cast<uint8_t>(x));
          CHECK_EQ(static_cast<int>(got[1]), static_cast<uint8_t>(y));
        }
      }
    }
  }
}

TEST(aSinglePixelIsItsOwnRotation) {
  auto src = ramp(1, 1);
  std::vector<uint8_t> dst(src.size());
  rotate_quarter(src.data(), dst.data(), 1, 1, 4, true);
  CHECK(dst == src);
}

TEST(aSingleRowBecomesAColumn) {
  const int w = 4, h = 1;
  auto src = ramp(w, h);
  std::vector<uint8_t> dst(src.size());
  rotate_quarter(src.data(), dst.data(), w, h, 4, true);
  // Destination is 1 wide and w tall; the row reads down it in order.
  for (int x = 0; x < w; ++x) {
    CHECK_EQ(static_cast<int>(at(dst, 1, 0, x)[0]), x);
  }
}

TEST(everyPixelIsAccountedFor) {
  // Nothing dropped and nothing written twice: the destination must be a permutation.
  const int w = 13, h = 5;
  auto src = ramp(w, h);
  std::vector<uint8_t> dst(src.size(), 0xAB);
  rotate_quarter(src.data(), dst.data(), w, h, 4, true);
  int untouched = 0;
  for (int i = 0; i < static_cast<int>(dst.size()); i += 4) {
    if (dst[i + 3] != 255) ++untouched;
  }
  CHECK_EQ(untouched, 0);
}

TEST(sixteenBitPixelsMoveWhole) {
  // RGBA half-float is 8 bytes a pixel, which the byte-agnostic copy has to carry intact.
  const int w = 3, h = 2, bpp = 8;
  std::vector<uint8_t> src(static_cast<size_t>(w) * h * bpp);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint8_t>(i);
  std::vector<uint8_t> dst(src.size());
  rotate_quarter(src.data(), dst.data(), w, h, bpp, true);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t *from = src.data() + (static_cast<size_t>(y) * w + x) * bpp;
      const uint8_t *to = dst.data() + (static_cast<size_t>(x) * h + (h - 1 - y)) * bpp;
      for (int b = 0; b < bpp; ++b) CHECK_EQ(static_cast<int>(to[b]), static_cast<int>(from[b]));
    }
  }
}

int main() { return check::main(); }
