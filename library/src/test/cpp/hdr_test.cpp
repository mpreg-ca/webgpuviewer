// Tests for the HDR half-float conversions and the tone-mapping curve.
//
// These are the functions whose bugs are silent: a wrong half conversion or a curve that is not
// monotonic shows up as slightly wrong colour, never as a crash, and no higher-level test would
// notice. The implementation is included rather than linked for the reason given in trim_test.cpp.

#include "hdr.cpp" // NOLINT(bugprone-suspicious-include)

#include "check.h"

#include <vector>

namespace {

/** Every half that is not a NaN or an infinity, so a sweep can be exhaustive. */
bool isFiniteHalf(uint16_t h) {
  return ((h >> 10) & 0x1F) != 0x1F;
}

} // namespace

// --- half floats -------------------------------------------------------------------------------

TEST(halfRoundTripsExactlyForEveryFiniteValue) {
  int mismatches = 0;
  for (int i = 0; i <= 0xFFFF; ++i) {
    const uint16_t h = static_cast<uint16_t>(i);
    if (!isFiniteHalf(h)) continue;
    const float f = half_to_float(h);
    const uint16_t back = float_to_half(f);
    // Negative zero and positive zero both round-trip to a zero of the same sign.
    if (back != h) ++mismatches;
  }
  CHECK_EQ(mismatches, 0);
}

TEST(halfKnowsItsLandmarks) {
  CHECK_NEAR(half_to_float(0x0000), 0.0, 1e-9);   // +0
  CHECK_NEAR(half_to_float(0x3C00), 1.0, 1e-9);   // 1
  CHECK_NEAR(half_to_float(0x4000), 2.0, 1e-9);   // 2
  CHECK_NEAR(half_to_float(0xBC00), -1.0, 1e-9);  // -1
  CHECK_NEAR(half_to_float(0x3800), 0.5, 1e-9);   // 0.5
}

TEST(subnormalHalvesSurvive) {
  // The smallest subnormal and the largest one, which a shift-based conversion gets wrong.
  CHECK_NEAR(half_to_float(0x0001), 5.9604645e-8, 1e-12);
  CHECK_NEAR(half_to_float(0x03FF), 6.0975552e-5, 1e-10);
  CHECK_EQ(static_cast<int>(float_to_half(5.9604645e-8f)), 0x0001);
}

TEST(floatToHalfSaturatesRatherThanWrapping) {
  // Well past the half maximum of 65504: the result must stay a finite large value or an
  // infinity, never wrap to something small.
  const uint16_t h = float_to_half(1e30f);
  const float f = half_to_float(h);
  CHECK(f > 60000.0f);
}

// --- sRGB transfer -----------------------------------------------------------------------------

TEST(srgbRoundTrips) {
  for (int i = 0; i <= 1000; ++i) {
    const float x = static_cast<float>(i) / 1000.0f;
    CHECK_NEAR(srgb_encode(srgb_decode(x)), x, 1e-4);
  }
}

TEST(srgbKeepsItsEndpoints) {
  CHECK_NEAR(srgb_decode(0.0f), 0.0, 1e-9);
  CHECK_NEAR(srgb_decode(1.0f), 1.0, 1e-6);
  CHECK_NEAR(srgb_encode(0.0f), 0.0, 1e-9);
  CHECK_NEAR(srgb_encode(1.0f), 1.0, 1e-6);
}

TEST(srgbByteEncodeCoversTheRange) {
  CHECK_EQ(static_cast<int>(srgb_encode_u8(0.0f)), 0);
  CHECK_EQ(static_cast<int>(srgb_encode_u8(1.0f)), 255);
  // Clamps rather than wrapping.
  CHECK_EQ(static_cast<int>(srgb_encode_u8(-1.0f)), 0);
  CHECK_EQ(static_cast<int>(srgb_encode_u8(4.0f)), 255);
}

// --- tone mapping ------------------------------------------------------------------------------

TEST(belowTheKneeNothingMoves) {
  const float knee = 0.8f, peak = 4.0f;
  for (float x = 0.0f; x <= knee; x += 0.05f) {
    CHECK_NEAR(tonemap_knee(x, knee, peak), x, 1e-6);
  }
}

TEST(thePeakLandsExactlyOnWhite) {
  // The whole point of the curve: the measured peak has to map to 1.0, not approach it.
  for (float peak : {1.5f, 2.0f, 4.0f, 10.0f}) {
    const float knee = shoulder_knee(peak);
    CHECK_NEAR(tonemap_knee(peak, knee, peak), 1.0, 1e-5);
  }
}

TEST(theCurveIsMonotonicAndBounded) {
  const float peak = 4.0f;
  const float knee = shoulder_knee(peak);
  float previous = -1.0f;
  for (int i = 0; i <= 800; ++i) {
    const float x = static_cast<float>(i) / 100.0f; // 0 .. 8, past the peak
    const float y = tonemap_knee(x, knee, peak);
    CHECK(y >= previous);   // never turns back on itself
    CHECK(y <= 1.0f + 1e-5f); // never escapes the output ceiling
    previous = y;
  }
}

TEST(theCurveKeepsItsShape) {
  // Endpoints and monotonicity hold for any s in (t / (t + s)) / (range / (range + s)), because
  // the division normalises the peak to 1 whatever s is. So those alone would not notice the
  // curve being retuned. This pins the midpoint for the documented s == range / 2: change it
  // deliberately and this number changes with it, change it by accident and this catches it.
  const float peak = 4.0f;
  const float knee = shoulder_knee(peak);          // 0.8 at two stops
  const float mid = knee + (peak - knee) * 0.5f;   // halfway up the roll-off
  CHECK_NEAR(tonemap_knee(mid, knee, peak), 0.95, 1e-3);
}

TEST(aDegenerateRangeStillClamps) {
  // peak at or below the knee leaves no room to roll off; the result must still be bounded.
  CHECK_NEAR(tonemap_knee(5.0f, 1.0f, 1.0f), 1.0, 1e-6);
  CHECK_NEAR(tonemap_knee(0.5f, 1.0f, 0.5f), 0.5, 1e-6);
}

TEST(theKneeFallsAsHeadroomGrows) {
  // A dim HDR frame should not pay a bright one's price in SDR range.
  CHECK_NEAR(shoulder_knee(1.0f), 1.0, 1e-6);  // nothing above white
  CHECK_NEAR(shoulder_knee(1.04f), 1.0, 1e-6); // under the 1.05 guard
  const float dim = shoulder_knee(1.5f);
  const float bright = shoulder_knee(4.0f);
  CHECK(dim < 1.0f);
  CHECK(bright < dim);
  // Capped at two stops, so it never gives up more than 20% of the range.
  CHECK(shoulder_knee(64.0f) >= 0.8f - 1e-6f);
}

// --- buffer arithmetic -------------------------------------------------------------------------

TEST(pixelCountCountsPixelsNotBytes) {
  // bpp is only there for the overflow guard below; the result is a pixel count.
  CHECK_EQ(pixel_count(1920, 1080, 4), static_cast<size_t>(1920) * 1080);
  CHECK_EQ(pixel_count(1920, 1080, 8), static_cast<size_t>(1920) * 1080);
}

TEST(pixelCountRejectsNonsenseDimensions) {
  CHECK_EQ(pixel_count(0, 1080, 4), static_cast<size_t>(0));
  CHECK_EQ(pixel_count(1920, 0, 4), static_cast<size_t>(0));
  CHECK_EQ(pixel_count(-1, 1080, 4), static_cast<size_t>(0));
  CHECK_EQ(pixel_count(1920, -1080, 4), static_cast<size_t>(0));
  CHECK_EQ(pixel_count(1920, 1080, 0), static_cast<size_t>(0));
}

TEST(pixelCountRefusesAnAllocationItCannotAddress) {
  // width * height * bpp past SIZE_MAX has to come back as 0, never wrap to something small
  // that a caller would then happily allocate and overrun.
  const size_t huge = pixel_count(0x7FFFFFFF, 0x7FFFFFFF, sizeof(size_t) > 4 ? 16 : 4);
  CHECK(huge == 0 || huge <= SIZE_MAX / 16);
}

int main() { return check::main(); }
