// Distance kernels and the three metric policies (L2, InnerProduct, Cosine).
//
// Every policy exposes the same static interface so that indexes can be templated on it:
//   static constexpr Metric kMetric;
//   static float distance(const float* a, const float* b, std::size_t d) noexcept;  // smaller = closer
//   static void  check(const float* v, std::size_t d);    // throws if the vector is unusable
//   static void  prepare(float* v, std::size_t d);        // in-place canonicalisation (cosine: normalise)
//
// Distances are symmetric bit-for-bit (distance(a,b) == distance(b,a)), which the brute-force
// reference and the graph index rely on to produce identical tie-breaking.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "hnsw/common.h"

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
#include <arm_neon.h>
#define HNSW_SIMD_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define HNSW_SIMD_AVX2 1
#endif

namespace hnsw {

enum class Metric : std::uint8_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

const char* metric_name(Metric m) noexcept;
/// Accepts "l2", "ip"/"inner_product", "cosine"/"cos". Throws InvalidArgument otherwise.
Metric parse_metric(const std::string& name);

namespace kernels {

/// Plain reference loops. Without -ffast-math the compiler may not reassociate the float
/// reduction, so these stay genuinely scalar; they are the numerical reference for tests.
inline float l2_sqr_scalar(const float* a, const float* b, std::size_t d) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < d; ++i) {
    const float t = a[i] - b[i];
    sum += t * t;
  }
  return sum;
}

inline float dot_scalar(const float* a, const float* b, std::size_t d) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < d; ++i) sum += a[i] * b[i];
  return sum;
}

#if defined(HNSW_SIMD_AVX2)
inline float hsum256(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
inline __m256 fmadd256(__m256 a, __m256 b, __m256 acc) noexcept {
#if defined(__FMA__)
  return _mm256_fmadd_ps(a, b, acc);
#else
  return _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#endif
}
#endif

/// Vectorised squared-L2 (NEON on arm64, AVX2 on x86-64, 4-way unrolled scalar otherwise).
inline float l2_sqr_simd(const float* a, const float* b, std::size_t d) noexcept {
  std::size_t i = 0;
#if defined(HNSW_SIMD_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
  for (; i + 16 <= d; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  for (; i + 4 <= d; i += 4) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    acc0 = vfmaq_f32(acc0, d0, d0);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#elif defined(HNSW_SIMD_AVX2)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= d; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = fmadd256(d0, d0, acc0);
    acc1 = fmadd256(d1, d1, acc1);
  }
  for (; i + 8 <= d; i += 8) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = fmadd256(d0, d0, acc0);
  }
  float sum = hsum256(_mm256_add_ps(acc0, acc1));
#else
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (; i + 4 <= d; i += 4) {
    const float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1];
    const float t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
    s0 += t0 * t0;
    s1 += t1 * t1;
    s2 += t2 * t2;
    s3 += t3 * t3;
  }
  float sum = (s0 + s1) + (s2 + s3);
#endif
  for (; i < d; ++i) {
    const float t = a[i] - b[i];
    sum += t * t;
  }
  return sum;
}

/// Vectorised dot product.
inline float dot_simd(const float* a, const float* b, std::size_t d) noexcept {
  std::size_t i = 0;
#if defined(HNSW_SIMD_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = acc0, acc2 = acc0, acc3 = acc0;
  for (; i + 16 <= d; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= d; i += 4) acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#elif defined(HNSW_SIMD_AVX2)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  for (; i + 16 <= d; i += 16) {
    acc0 = fmadd256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = fmadd256(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= d; i += 8) acc0 = fmadd256(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  float sum = hsum256(_mm256_add_ps(acc0, acc1));
#else
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  for (; i + 4 <= d; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  float sum = (s0 + s1) + (s2 + s3);
#endif
  for (; i < d; ++i) sum += a[i] * b[i];
  return sum;
}

/// Name of the SIMD path compiled into l2_sqr_simd/dot_simd.
const char* simd_name() noexcept;

/// The kernels the metric policies use. Define HNSW_FORCE_SCALAR to build a scalar-only library.
inline float l2_sqr(const float* a, const float* b, std::size_t d) noexcept {
#if defined(HNSW_FORCE_SCALAR)
  return l2_sqr_scalar(a, b, d);
#else
  return l2_sqr_simd(a, b, d);
#endif
}
inline float dot(const float* a, const float* b, std::size_t d) noexcept {
#if defined(HNSW_FORCE_SCALAR)
  return dot_scalar(a, b, d);
#else
  return dot_simd(a, b, d);
#endif
}

}  // namespace kernels

/// Squared Euclidean distance (monotone in L2, so rankings are identical and no sqrt is paid).
struct L2 {
  static constexpr Metric kMetric = Metric::L2;
  static constexpr bool kIsMetric = true;  // satisfies the triangle inequality (on sqrt)
  static float distance(const float* a, const float* b, std::size_t d) noexcept {
    return kernels::l2_sqr(a, b, d);
  }
  static void check(const float* /*v*/, std::size_t /*d*/) {}
  static void prepare(float* /*v*/, std::size_t /*d*/) {}
};

/// Maximum inner product search expressed as a distance: -<a, b>. Not a metric (no triangle
/// inequality), which the experiments discuss.
struct InnerProduct {
  static constexpr Metric kMetric = Metric::InnerProduct;
  /// d(a, a) is not minimal and large-norm vectors are "close" to everything, which breaks the
  /// geometric assumption behind HNSW's neighbour-diversity heuristic (see HnswParams::keep_pruned).
  static constexpr bool kIsMetric = false;
  static float distance(const float* a, const float* b, std::size_t d) noexcept {
    return -kernels::dot(a, b, d);
  }
  static void check(const float* /*v*/, std::size_t /*d*/) {}
  static void prepare(float* /*v*/, std::size_t /*d*/) {}
};

/// Cosine distance 1 - cos(a, b). Vectors are L2-normalised once on insertion / query, so the
/// hot loop is a single dot product. The zero vector has no direction and is rejected.
struct Cosine {
  static constexpr Metric kMetric = Metric::Cosine;
  static constexpr bool kIsMetric = true;  // angular order on the unit sphere, equivalent to L2 there
  static float distance(const float* a, const float* b, std::size_t d) noexcept {
    return 1.0f - kernels::dot(a, b, d);
  }
  static double norm2(const float* v, std::size_t d) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < d; ++i) s += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    return s;
  }
  static void check(const float* v, std::size_t d) {
    if (!(norm2(v, d) > 0.0)) {
      throw InvalidArgument("zero vector has no direction and cannot be used with the cosine metric");
    }
  }
  static void prepare(float* v, std::size_t d) {
    const double n2 = norm2(v, d);
    if (!(n2 > 0.0)) {
      throw InvalidArgument("zero vector has no direction and cannot be used with the cosine metric");
    }
    const double inv = 1.0 / std::sqrt(n2);
    for (std::size_t i = 0; i < d; ++i) v[i] = static_cast<float>(static_cast<double>(v[i]) * inv);
  }
};

}  // namespace hnsw
