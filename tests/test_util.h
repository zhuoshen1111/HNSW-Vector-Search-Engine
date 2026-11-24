// Shared helpers for the unit tests: seeded data, reference answers and comparisons.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "hnsw/brute_force_index.h"
#include "hnsw/common.h"
#include "hnsw/hnsw_index.h"
#include "hnsw/index_factory.h"

namespace testutil {

/// Seeded row-major dataset: `clusters` Gaussian blobs (or uniform when clusters == 0).
/// Cluster centres come from `center_seed` (default: seed), so queries drawn with another `seed`
/// but the same `center_seed` follow the data distribution.
inline std::vector<float> make_data(std::size_t n, std::size_t d, std::uint64_t seed, std::size_t clusters = 8,
                                    std::uint64_t center_seed = 0) {
  std::mt19937_64 crng(center_seed == 0 ? seed : center_seed);
  std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  std::normal_distribution<float> gauss(0.0f, 0.05f);
  std::vector<float> centers(std::max<std::size_t>(clusters, 1) * d);
  for (float& c : centers) c = uni(crng);
  std::vector<float> out(n * d);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = clusters == 0 ? 0 : static_cast<std::size_t>(rng() % clusters);
    for (std::size_t j = 0; j < d; ++j) out[i * d + j] = clusters == 0 ? uni(rng) : centers[c * d + j] + gauss(rng);
  }
  return out;
}

/// Queries from the same mixture as make_data(n, d, data_seed, clusters).
inline std::vector<float> make_queries(std::size_t nq, std::size_t d, std::uint64_t data_seed, std::size_t clusters = 8) {
  return make_data(nq, d, data_seed + 1000, clusters, data_seed);
}

/// Independent linear-scan reference: sorts every (distance, id) pair with std::sort.
template <typename Dist>
std::vector<hnsw::SearchResult> linear_scan(const std::vector<float>& data, const std::vector<hnsw::label_t>& ids,
                                            std::size_t d, const float* q, std::size_t k) {
  std::vector<float> qq(q, q + d);
  Dist::prepare(qq.data(), d);
  std::vector<hnsw::SearchResult> all;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::vector<float> v(data.begin() + static_cast<std::ptrdiff_t>(i * d),
                         data.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
    Dist::prepare(v.data(), d);
    all.push_back({ids[i], Dist::distance(qq.data(), v.data(), d)});
  }
  std::sort(all.begin(), all.end(), hnsw::ResultOrder{});
  if (all.size() > k) all.resize(k);
  return all;
}

inline double overlap(const std::vector<hnsw::SearchResult>& got, const std::vector<hnsw::SearchResult>& truth) {
  if (truth.empty()) return got.empty() ? 1.0 : 0.0;
  std::size_t hit = 0;
  for (const auto& g : got) {
    for (const auto& t : truth) {
      if (g.id == t.id) {
        ++hit;
        break;
      }
    }
  }
  return static_cast<double>(hit) / static_cast<double>(truth.size());
}

inline bool sorted_by_order(const std::vector<hnsw::SearchResult>& r) {
  return std::is_sorted(r.begin(), r.end(), hnsw::ResultOrder{});
}

inline hnsw::HnswParams small_params(std::size_t M = 8, std::size_t efc = 100) {
  hnsw::HnswParams p;
  p.M = M;
  p.ef_construction = efc;
  p.seed = 7;
  return p;
}

}  // namespace testutil
