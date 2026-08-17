// CPU implementations of the trim and background-detection
//
// Pixel layout is RGBA8, row-major, tightly packed: byte 0 is red, byte 3 is
// alpha. That matches both TextureFormat.RGBA8Unorm and Android's ARGB_8888
// bitmaps, which are RGBA in memory order despite the name. Bytes are read
// individually rather than through a uint32_t so the channel identity does not
// depend on endianness.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <jni.h>
#include <thread>
#include <vector>

namespace {

constexpr int kChannels = 4;

// ---------------------------------------------------------------------------
// Trim
// ---------------------------------------------------------------------------

/**
 * Foreground test for one background colour, matching is_foreground in the WGSL
 * trim shaders.
 *
 * The shader composites the pixel over the background before comparing:
 *   diff = |rgb * a + bg * (1 - a) - bg| = a * |rgb - bg|
 * so the blend collapses to a single multiply by alpha. Everything is kept in
 * 0..255 units to avoid normalising every pixel: the shader's
 *   a01 * |c01 - bg01| > threshold
 * scales to
 *   a * |c - bg255| > threshold * 255 * 255
 */
struct ColorTest {
  float bg255[3];
  float thresholdScaled;
  // Fully opaque pixels dominate real pages, and for those the alpha multiply
  // drops out, leaving a comparison that depends only on the byte value.
  uint8_t opaqueForeground[3][256];
};

ColorTest makeColorTest(float r, float g, float b, float threshold) {
  ColorTest test{};
  test.bg255[0] = r * 255.0f;
  test.bg255[1] = g * 255.0f;
  test.bg255[2] = b * 255.0f;
  test.thresholdScaled = threshold * 255.0f * 255.0f;

  const float opaqueThreshold = threshold * 255.0f;
  for (int ch = 0; ch < 3; ++ch) {
    for (int c = 0; c < 256; ++c) {
      test.opaqueForeground[ch][c] =
          std::fabs(static_cast<float>(c) - test.bg255[ch]) > opaqueThreshold;
    }
  }
  return test;
}

inline bool isForeground(const ColorTest &test, const uint8_t *px) {
  const uint8_t a = px[3];
  if (a == 255) {
    return (test.opaqueForeground[0][px[0]] | test.opaqueForeground[1][px[1]] |
            test.opaqueForeground[2][px[2]]) != 0;
  }
  const float af = static_cast<float>(a);
  return af * std::fabs(static_cast<float>(px[0]) - test.bg255[0]) >
             test.thresholdScaled ||
         af * std::fabs(static_cast<float>(px[1]) - test.bg255[1]) >
             test.thresholdScaled ||
         af * std::fabs(static_cast<float>(px[2]) - test.bg255[2]) >
             test.thresholdScaled;
}

/**
 * Bounding box of the foreground pixels.
 *
 * Seeded the way the shader seeds its result buffer - min at the image extent,
 * max at zero - so an image with no foreground at all produces the same
 * (width, height, 0, 0) the GPU path produces, and the callers that already
 * handle that degenerate result keep working.
 */
struct Bounds {
  int32_t minX, minY, maxX, maxY;

  void reset(int width, int height) {
    minX = width;
    minY = height;
    maxX = 0;
    maxY = 0;
  }

  void merge(const Bounds &other) {
    minX = std::min(minX, other.minX);
    minY = std::min(minY, other.minY);
    maxX = std::max(maxX, other.maxX);
    maxY = std::max(maxY, other.maxY);
  }
};

/**
 * Accumulate bounds for every colour over rows [y0, y1).
 *
 * Colours are the inner loop so a row is walked while it is still in cache.
 * Within a row, the scan runs inward from both ends and stops at the first hit:
 * a page with margins costs two short scans, and only a row that is entirely
 * background has to be read end to end (there is no way to prove it empty
 * otherwise).
 */
void scanBand(const uint8_t *pixels, int width, int y0, int y1,
              const ColorTest *tests, int colorCount, Bounds *bounds) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t *row = pixels + static_cast<size_t>(y) * width * kChannels;

    for (int ci = 0; ci < colorCount; ++ci) {
      const ColorTest &test = tests[ci];
      Bounds &bb = bounds[ci];

      int first = -1;
      for (int x = 0; x < width; ++x) {
        if (isForeground(test, row + static_cast<size_t>(x) * kChannels)) {
          first = x;
          break;
        }
      }
      if (first < 0) {
        continue; // Row is entirely background for this colour.
      }

      int last = first;
      for (int x = width - 1; x > first; --x) {
        if (isForeground(test, row + static_cast<size_t>(x) * kChannels)) {
          last = x;
          break;
        }
      }

      if (first < bb.minX)
        bb.minX = first;
      if (last > bb.maxX)
        bb.maxX = last;
      if (y < bb.minY)
        bb.minY = y;
      if (y > bb.maxY)
        bb.maxY = y;
    }
  }
}

int chooseThreadCount(int width, int height) {
  // Below roughly a quarter-megapixel the scan is short enough that spawning
  // threads costs more than it saves.
  if (static_cast<int64_t>(width) * height < 256 * 1024) {
    return 1;
  }
  unsigned hardware = std::thread::hardware_concurrency();
  int count = static_cast<int>(std::min(hardware ? hardware : 1u, 8u));
  // Keep bands big enough to be worth a thread.
  count = std::min(count, height / 64);
  return std::max(1, count);
}

// ---------------------------------------------------------------------------
// Background detection
// ---------------------------------------------------------------------------

struct EdgeAccum {
  double sum[3];
  double sumSq[3];
  int64_t total;
};

inline void accumulate(EdgeAccum &acc, const uint8_t *px) {
  // Alpha is deliberately ignored, matching the edge-detect shader, which reads
  // .rgb straight out of the texture without compositing.
  for (int ch = 0; ch < 3; ++ch) {
    const double v = px[ch] / 255.0;
    acc.sum[ch] += v;
    acc.sumSq[ch] += v * v;
  }
  acc.total++;
}

/**
 * An edge counts as solid when no channel varies more than threshold^2, in
 * which case its mean colour is written to outRgb as 0xRRGGBB.
 *
 * The GPU path evaluates variance per 4096px tile and requires every tile on
 * the edge to be solid; here the edge is one run, so the variance covers the
 * whole edge. The two agree for any image that fits in a single tile, and for
 * larger ones whole-edge variance is the stricter, more meaningful test.
 */
bool solidEdgeColor(const EdgeAccum &acc, float threshold, int &outRgb) {
  if (acc.total == 0) {
    return false;
  }
  const double inv = 1.0 / static_cast<double>(acc.total);
  double maxVar = 0.0;
  int rgb = 0;

  for (int ch = 0; ch < 3; ++ch) {
    const double avg = acc.sum[ch] * inv;
    // var = E[X^2] - E[X]^2
    const double var = acc.sumSq[ch] * inv - avg * avg;
    if (var > maxVar) {
      maxVar = var;
    }
    const int v = std::clamp(static_cast<int>(avg * 255.0), 0, 255);
    rgb = (rgb << 8) | v;
  }

  if (maxVar >= static_cast<double>(threshold) * threshold) {
    return false;
  }
  outRgb = rgb;
  return true;
}

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_ca_mpreg_webgpuviewer_TrimNative_findTrim(JNIEnv *env, jobject thiz,
                                               jobject pixelBuffer, jint width,
                                               jint height, jfloatArray colors,
                                               jfloat threshold,
                                               jintArray outBounds) {
  if (width <= 0 || height <= 0) {
    return JNI_FALSE;
  }

  const uint8_t *pixels =
      static_cast<const uint8_t *>(env->GetDirectBufferAddress(pixelBuffer));
  if (pixels == nullptr) {
    return JNI_FALSE;
  }
  const jlong capacity = env->GetDirectBufferCapacity(pixelBuffer);
  if (capacity < static_cast<jlong>(width) * height * kChannels) {
    return JNI_FALSE;
  }

  const jsize colorFloats = env->GetArrayLength(colors);
  const int colorCount = colorFloats / 3;
  if (colorCount <= 0 || env->GetArrayLength(outBounds) < colorCount * 4) {
    return JNI_FALSE;
  }

  jfloat *colorData = env->GetFloatArrayElements(colors, nullptr);
  if (colorData == nullptr) {
    return JNI_FALSE;
  }

  std::vector<ColorTest> tests;
  tests.reserve(colorCount);
  for (int i = 0; i < colorCount; ++i) {
    tests.push_back(makeColorTest(colorData[i * 3], colorData[i * 3 + 1],
                                  colorData[i * 3 + 2], threshold));
  }
  env->ReleaseFloatArrayElements(colors, colorData, JNI_ABORT);

  std::vector<Bounds> merged(colorCount);
  for (int i = 0; i < colorCount; ++i) {
    merged[i].reset(width, height);
  }

  const int threadCount = chooseThreadCount(width, height);

  if (threadCount == 1) {
    scanBand(pixels, width, 0, height, tests.data(), colorCount, merged.data());
  } else {
    // Worker threads only read the direct buffer and their own band results, so
    // they never touch JNI and need no JNIEnv of their own.
    std::vector<std::vector<Bounds>> bandResults(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount - 1);

    const int rowsPerBand = (height + threadCount - 1) / threadCount;

    for (int t = 0; t < threadCount; ++t) {
      bandResults[t].resize(colorCount);
      for (int i = 0; i < colorCount; ++i) {
        bandResults[t][i].reset(width, height);
      }
    }

    for (int t = 1; t < threadCount; ++t) {
      const int y0 = std::min(t * rowsPerBand, height);
      const int y1 = std::min(y0 + rowsPerBand, height);
      if (y0 >= y1) {
        continue;
      }
      workers.emplace_back([=, &tests, &bandResults] {
        scanBand(pixels, width, y0, y1, tests.data(), colorCount,
                 bandResults[t].data());
      });
    }

    scanBand(pixels, width, 0, std::min(rowsPerBand, height), tests.data(),
             colorCount, bandResults[0].data());

    for (auto &worker : workers) {
      worker.join();
    }

    for (int t = 0; t < threadCount; ++t) {
      for (int i = 0; i < colorCount; ++i) {
        merged[i].merge(bandResults[t][i]);
      }
    }
  }

  std::vector<jint> out(colorCount * 4);
  for (int i = 0; i < colorCount; ++i) {
    out[i * 4 + 0] = merged[i].minX;
    out[i * 4 + 1] = merged[i].minY;
    out[i * 4 + 2] = merged[i].maxX;
    out[i * 4 + 3] = merged[i].maxY;
  }
  env->SetIntArrayRegion(outBounds, 0, colorCount * 4, out.data());

  return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_ca_mpreg_webgpuviewer_TrimNative_detectBackground(JNIEnv *env,
                                                       jobject thiz,
                                                       jobject pixelBuffer,
                                                       jint width, jint height,
                                                       jfloat threshold) {
  const jint kWhite = static_cast<jint>(0xFFFFFFFFu);

  if (width <= 0 || height <= 0) {
    return kWhite;
  }

  const uint8_t *pixels =
      static_cast<const uint8_t *>(env->GetDirectBufferAddress(pixelBuffer));
  if (pixels == nullptr) {
    return kWhite;
  }
  const jlong capacity = env->GetDirectBufferCapacity(pixelBuffer);
  if (capacity < static_cast<jlong>(width) * height * kChannels) {
    return kWhite;
  }

  EdgeAccum left{}, right{}, top{}, bottom{};

  const size_t stride = static_cast<size_t>(width) * kChannels;
  for (int y = 0; y < height; ++y) {
    const uint8_t *row = pixels + static_cast<size_t>(y) * stride;
    accumulate(left, row);
    accumulate(right, row + static_cast<size_t>(width - 1) * kChannels);
  }

  const uint8_t *topRow = pixels;
  const uint8_t *bottomRow = pixels + static_cast<size_t>(height - 1) * stride;
  for (int x = 0; x < width; ++x) {
    accumulate(top, topRow + static_cast<size_t>(x) * kChannels);
    accumulate(bottom, bottomRow + static_cast<size_t>(x) * kChannels);
  }

  // Edge order matters: the "first solid edge wins" rule below resolves ties
  // the same way the GPU path does, which aggregates left, right, top, bottom.
  const EdgeAccum *edges[4] = {&left, &right, &top, &bottom};

  int solid[4];
  int solidCount = 0;
  for (const EdgeAccum *edge : edges) {
    int rgb = 0;
    if (solidEdgeColor(*edge, threshold, rgb)) {
      solid[solidCount++] = rgb;
    }
  }

  // Rule: any white edge -> white.
  for (int i = 0; i < solidCount; ++i) {
    const int r = (solid[i] >> 16) & 0xFF;
    const int g = (solid[i] >> 8) & 0xFF;
    const int b = solid[i] & 0xFF;
    if (r >= 242 && g >= 242 && b >= 242) {
      return kWhite;
    }
  }

  // Rule: any colour edge -> that colour, opaque.
  if (solidCount > 0) {
    return static_cast<jint>(0xFF000000u | static_cast<uint32_t>(solid[0]));
  }

  // Rule: otherwise white.
  return kWhite;
}
