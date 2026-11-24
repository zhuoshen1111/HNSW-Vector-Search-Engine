#include "hnsw/distance.h"

#include <algorithm>
#include <cctype>

namespace hnsw {

const char* metric_name(Metric m) noexcept {
  switch (m) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

Metric parse_metric(const std::string& name) {
  std::string s = name;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (s == "l2" || s == "euclidean") return Metric::L2;
  if (s == "ip" || s == "inner_product" || s == "dot") return Metric::InnerProduct;
  if (s == "cosine" || s == "cos" || s == "angular") return Metric::Cosine;
  throw InvalidArgument("unknown metric '" + name + "' (expected l2, ip or cosine)");
}

namespace kernels {
const char* simd_name() noexcept {
#if defined(HNSW_SIMD_NEON)
  return "neon";
#elif defined(HNSW_SIMD_AVX2)
#if defined(__FMA__)
  return "avx2+fma";
#else
  return "avx2";
#endif
#else
  return "unrolled-scalar";
#endif
}
}  // namespace kernels

}  // namespace hnsw
