// Tests for the CPU trim and background detection.
//
// The implementation's helpers live in an anonymous namespace, so the translation unit is included
// rather than linked. That is deliberate: these functions are the contract that has to keep
// matching the WGSL shaders, and testing them through the JNI entry points would need a JVM.

#include "trim.cpp" // NOLINT(bugprone-suspicious-include)

#include "check.h"

#include <vector>

namespace {

constexpr float kThreshold = 0.05f;

/** A solid RGBA8 image, so a test only has to draw what it cares about. */
std::vector<uint8_t> filled(int width, int height, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  std::vector<uint8_t> px(static_cast<size_t>(width) * height * 4);
  for (size_t i = 0; i < px.size(); i += 4) {
    px[i] = r;
    px[i + 1] = g;
    px[i + 2] = b;
    px[i + 3] = a;
  }
  return px;
}

void setPixel(std::vector<uint8_t> &px, int width, int x, int y,
              uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  const size_t i = (static_cast<size_t>(y) * width + x) * 4;
  px[i] = r;
  px[i + 1] = g;
  px[i + 2] = b;
  px[i + 3] = a;
}

ColorTest white() { return makeColorTest(1.0f, 1.0f, 1.0f, kThreshold); }

} // namespace

// --- makeColorTest -----------------------------------------------------------------------------

TEST(colorTestScalesIntoByteUnits) {
  const ColorTest test = makeColorTest(1.0f, 0.5f, 0.0f, kThreshold);
  CHECK_NEAR(test.bg255[0], 255.0, 1e-4);
  CHECK_NEAR(test.bg255[1], 127.5, 1e-4);
  CHECK_NEAR(test.bg255[2], 0.0, 1e-4);
  // The shader compares a01 * |c01 - bg01| > threshold, which scales by 255 twice.
  CHECK_NEAR(test.thresholdScaled, kThreshold * 255.0 * 255.0, 1e-3);
}

TEST(opaqueLookupMatchesTheDocumentedRule) {
  const ColorTest test = white();
  const double cutoff = kThreshold * 255.0;
  for (int c = 0; c < 256; ++c) {
    const bool expected = std::fabs(static_cast<double>(c) - 255.0) > cutoff;
    CHECK_EQ(static_cast<int>(test.opaqueForeground[0][c]), static_cast<int>(expected));
  }
}

// --- isForeground ------------------------------------------------------------------------------

TEST(backgroundItselfIsNeverForeground) {
  const ColorTest test = white();
  const uint8_t px[4] = {255, 255, 255, 255};
  CHECK(!isForeground(test, px));
}

TEST(aClearlyDifferentOpaquePixelIsForeground) {
  const ColorTest test = white();
  const uint8_t black[4] = {0, 0, 0, 255};
  CHECK(isForeground(test, black));
}

TEST(oneChannelIsEnoughToCount) {
  const ColorTest test = white();
  const uint8_t red[4] = {255, 0, 0, 255};
  CHECK(isForeground(test, red));
}

TEST(transparencyWeighsTheDifference) {
  const ColorTest test = white();
  // Black against white is the largest possible difference, so alpha alone decides. The rule is
  // a * |c - bg| > threshold * 255 * 255, i.e. a > threshold * 255 for this pixel.
  const uint8_t barelyThere[4] = {0, 0, 0, 12}; // 12 < 0.05 * 255 = 12.75
  CHECK(!isForeground(test, barelyThere));
  const uint8_t justEnough[4] = {0, 0, 0, 13};
  CHECK(isForeground(test, justEnough));
}

TEST(fullyTransparentIsNeverForeground) {
  const ColorTest test = white();
  const uint8_t invisible[4] = {0, 0, 0, 0};
  CHECK(!isForeground(test, invisible));
}

// --- scanBand ----------------------------------------------------------------------------------

TEST(scanFindsTheBoxAroundContent) {
  const int w = 32, h = 24;
  std::vector<uint8_t> px = filled(w, h, 255, 255, 255);
  for (int y = 6; y <= 17; ++y) {
    for (int x = 4; x <= 20; ++x) setPixel(px, w, x, y, 0, 0, 0);
  }

  ColorTest tests[1] = {white()};
  Bounds bounds[1];
  bounds[0].reset(w, h);
  scanBand(px.data(), w, 0, h, tests, 1, bounds);

  CHECK_EQ(bounds[0].minX, 4);
  CHECK_EQ(bounds[0].maxX, 20);
  CHECK_EQ(bounds[0].minY, 6);
  CHECK_EQ(bounds[0].maxY, 17);
}

TEST(anEmptyImageLeavesTheSeedUntouched) {
  const int w = 16, h = 16;
  const std::vector<uint8_t> px = filled(w, h, 255, 255, 255);

  ColorTest tests[1] = {white()};
  Bounds bounds[1];
  bounds[0].reset(w, h);
  scanBand(px.data(), w, 0, h, tests, 1, bounds);

  // Seeded min at the extent and max at zero, so "no foreground" stays inverted and the caller
  // can tell it apart from a real box.
  CHECK_EQ(bounds[0].minX, w);
  CHECK_EQ(bounds[0].minY, h);
  CHECK_EQ(bounds[0].maxX, 0);
  CHECK_EQ(bounds[0].maxY, 0);
}

TEST(aSinglePixelBoundsItself) {
  const int w = 16, h = 16;
  std::vector<uint8_t> px = filled(w, h, 255, 255, 255);
  setPixel(px, w, 9, 3, 0, 0, 0);

  ColorTest tests[1] = {white()};
  Bounds bounds[1];
  bounds[0].reset(w, h);
  scanBand(px.data(), w, 0, h, tests, 1, bounds);

  CHECK_EQ(bounds[0].minX, 9);
  CHECK_EQ(bounds[0].maxX, 9);
  CHECK_EQ(bounds[0].minY, 3);
  CHECK_EQ(bounds[0].maxY, 3);
}

TEST(bandsAccumulateIntoTheSameBox) {
  const int w = 16, h = 32;
  std::vector<uint8_t> px = filled(w, h, 255, 255, 255);
  setPixel(px, w, 2, 1, 0, 0, 0);
  setPixel(px, w, 13, 30, 0, 0, 0);

  ColorTest tests[1] = {white()};
  Bounds bounds[1];
  bounds[0].reset(w, h);
  // Split exactly as the threaded path does.
  scanBand(px.data(), w, 0, 16, tests, 1, bounds);
  scanBand(px.data(), w, 16, h, tests, 1, bounds);

  CHECK_EQ(bounds[0].minX, 2);
  CHECK_EQ(bounds[0].maxX, 13);
  CHECK_EQ(bounds[0].minY, 1);
  CHECK_EQ(bounds[0].maxY, 30);
}

TEST(eachColourGetsItsOwnBox) {
  const int w = 16, h = 16;
  // A red field: black is foreground against red, and red is not.
  std::vector<uint8_t> px = filled(w, h, 255, 0, 0);
  setPixel(px, w, 5, 5, 0, 0, 0);

  ColorTest tests[2] = {makeColorTest(1.0f, 0.0f, 0.0f, kThreshold), white()};
  Bounds bounds[2];
  bounds[0].reset(w, h);
  bounds[1].reset(w, h);
  scanBand(px.data(), w, 0, h, tests, 2, bounds);

  // Against red, only the black pixel stands out.
  CHECK_EQ(bounds[0].minX, 5);
  CHECK_EQ(bounds[0].maxX, 5);
  // Against white, the whole red field does.
  CHECK_EQ(bounds[1].minX, 0);
  CHECK_EQ(bounds[1].maxX, w - 1);
}

// --- chooseThreadCount -------------------------------------------------------------------------

TEST(smallImagesStaySingleThreaded) {
  CHECK_EQ(chooseThreadCount(64, 64), 1);
  // Just under a quarter of a megapixel.
  CHECK_EQ(chooseThreadCount(512, 511), 1);
}

TEST(threadCountIsAlwaysUsable) {
  const int counts[][2] = {{2000, 3000}, {4000, 6000}, {512, 512}, {1, 1}, {8000, 40}};
  for (const auto &wh : counts) {
    const int n = chooseThreadCount(wh[0], wh[1]);
    CHECK(n >= 1);
    CHECK(n <= 8);
    // A band has to be worth a thread.
    if (n > 1) CHECK(wh[1] / n >= 64);
  }
}

// --- colour conversion -------------------------------------------------------------------------

TEST(srgbRoundTripsThroughLinear) {
  for (int b = 0; b <= 255; ++b) {
    const double linear = srgbByteToLinear(static_cast<uint8_t>(b));
    const double back = linearToSrgb(linear) * 255.0;
    CHECK_NEAR(back, static_cast<double>(b), 0.5);
  }
}

TEST(theLinearCurveKeepsItsEndpoints) {
  CHECK_NEAR(srgbByteToLinear(0), 0.0, 1e-9);
  CHECK_NEAR(srgbByteToLinear(255), 1.0, 1e-9);
  CHECK_NEAR(linearToSrgb(0.0), 0.0, 1e-9);
  CHECK_NEAR(linearToSrgb(1.0), 1.0, 1e-9);
}

// --- classifyEdge ------------------------------------------------------------------------------

TEST(aSolidEdgeIsReportedSolid) {
  const int count = 64;
  const std::vector<uint8_t> px = filled(count, 1, 255, 255, 255);
  const EdgeLine line{px.data(), 4, count};

  const EdgeResult result = classifyEdge(line);
  CHECK(result.solid);
  CHECK(result.isWhite);
}

TEST(aSolidColouredEdgeReportsItsColour) {
  const int count = 64;
  const std::vector<uint8_t> px = filled(count, 1, 0, 0, 0);
  const EdgeLine line{px.data(), 4, count};

  const EdgeResult result = classifyEdge(line);
  CHECK(result.solid);
  CHECK(!result.isWhite);
  CHECK_NEAR(result.linearMean[0], 0.0, 1e-6);
}

TEST(aNoisyEdgeIsNotSolid) {
  const int count = 64;
  std::vector<uint8_t> px = filled(count, 1, 0, 0, 0);
  // Half black, half white: no single colour describes this edge.
  for (int i = count / 2; i < count; ++i) {
    px[static_cast<size_t>(i) * 4 + 0] = 255;
    px[static_cast<size_t>(i) * 4 + 1] = 255;
    px[static_cast<size_t>(i) * 4 + 2] = 255;
  }
  const EdgeLine line{px.data(), 4, count};

  CHECK(!classifyEdge(line).solid);
}

TEST(strideWalksColumnsAsWellAsRows) {
  // A column of an image is the same line with a row-sized stride.
  const int w = 8, h = 16;
  const std::vector<uint8_t> px = filled(w, h, 255, 255, 255);
  const EdgeLine column{px.data(), static_cast<size_t>(w) * 4, h};

  const EdgeResult result = classifyEdge(column);
  CHECK(result.solid);
  CHECK(result.isWhite);
}

int main() { return check::main(); }
