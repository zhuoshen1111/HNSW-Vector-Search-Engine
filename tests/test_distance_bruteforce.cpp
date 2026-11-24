// Distance kernels, metric policies and the brute-force reference index.
#include <cmath>
#include <random>

#include "hnsw/brute_force_index.h"
#include "hnsw/distance.h"
#include "test_framework.h"
#include "test_util.h"

using namespace hnsw;

TEST(simd_kernels_match_scalar_reference) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  for (std::size_t d : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 100u, 128u, 960u, 1000u}) {
    std::vector<float> a(d), b(d);
    for (std::size_t i = 0; i < d; ++i) {
      a[i] = uni(rng);
      b[i] = uni(rng);
    }
    const double l2_ref = kernels::l2_sqr_scalar(a.data(), b.data(), d);
    const double dot_ref = kernels::dot_scalar(a.data(), b.data(), d);
    CHECK(std::abs(kernels::l2_sqr_simd(a.data(), b.data(), d) - l2_ref) <= 1e-4 * std::max(1.0, std::abs(l2_ref)));
    CHECK(std::abs(kernels::dot_simd(a.data(), b.data(), d) - dot_ref) <= 1e-4 * std::max(1.0, std::abs(dot_ref)));
    // Symmetry is exact (bit-for-bit), which keeps tie-breaking consistent between indexes.
    CHECK_EQ(L2::distance(a.data(), b.data(), d), L2::distance(b.data(), a.data(), d));
    CHECK_EQ(InnerProduct::distance(a.data(), b.data(), d), InnerProduct::distance(b.data(), a.data(), d));
    CHECK_EQ(L2::distance(a.data(), a.data(), d), 0.0f);
  }
}

TEST(metric_policies_known_values) {
  const float a[3] = {1, 0, 0};
  const float b[3] = {0, 2, 0};
  CHECK_EQ(L2::distance(a, b, 3), 5.0f);
  CHECK_EQ(InnerProduct::distance(a, b, 3), -0.0f);
  float c[3] = {3, 0, 4};
  Cosine::prepare(c, 3);
  CHECK(std::abs(c[0] - 0.6f) < 1e-6f && std::abs(c[2] - 0.8f) < 1e-6f);
  float x[3] = {2, 0, 0};
  Cosine::prepare(x, 3);
  CHECK(std::abs(Cosine::distance(x, c, 3) - 0.4f) < 1e-6f);
  float zero[3] = {0, 0, 0};
  CHECK_THROWS_AS(Cosine::prepare(zero, 3), InvalidArgument);
  CHECK_THROWS_AS(Cosine::check(zero, 3), InvalidArgument);
  CHECK_EQ(parse_metric("L2"), Metric::L2);
  CHECK_EQ(parse_metric("cosine"), Metric::Cosine);
  CHECK_EQ(parse_metric("ip"), Metric::InnerProduct);
  CHECK_THROWS_AS(parse_metric("manhattan"), InvalidArgument);
}

template <typename Dist>
void check_bruteforce_against_scan(std::uint64_t seed) {
  const std::size_t n = 600, d = 16;
  const auto data = testutil::make_data(n, d, seed);
  std::vector<label_t> ids(n);
  for (std::size_t i = 0; i < n; ++i) ids[i] = 1000 + 3 * i;
  for (TopKMethod m : {TopKMethod::QuickSelect, TopKMethod::Heap, TopKMethod::FullSort}) {
    BruteForceIndex<Dist> bf(d);
    bf.set_topk_method(m);
    for (std::size_t i = 0; i < n; ++i) bf.add(ids[i], data.data() + i * d, d);
    const auto queries = testutil::make_data(30, d, seed + 1);
    for (std::size_t q = 0; q < 30; ++q) {
      for (std::size_t k : {1u, 10u, 100u}) {
        const auto got = bf.knn(std::vector<float>(queries.begin() + q * d, queries.begin() + (q + 1) * d), k);
        const auto want = testutil::linear_scan<Dist>(data, ids, d, queries.data() + q * d, k);
        CHECK(got == want);
      }
    }
    CHECK(bf.check_invariants().empty());
  }
}

TEST(bruteforce_matches_linear_scan_all_metrics_and_topk_methods) {
  check_bruteforce_against_scan<L2>(21);
  check_bruteforce_against_scan<InnerProduct>(22);
  check_bruteforce_against_scan<Cosine>(23);
}

TEST(bruteforce_boundaries_and_errors) {
  BruteForceIndex<L2> bf(2);
  const std::vector<float> q = {0, 0};
  CHECK(bf.knn(q, 5).empty());  // empty index
  const float p[2] = {1, 1};
  bf.add(5, p, 2);
  CHECK(bf.knn(q, 0).empty());  // k = 0
  const auto one = bf.knn(q, 10);  // k > n
  REQUIRE(one.size() == 1);
  CHECK_EQ(one[0].id, 5u);
  CHECK_THROWS_AS(bf.add(5, p, 2), InvalidArgument);  // duplicate id
  const float p3[3] = {1, 2, 3};
  CHECK_THROWS_AS(bf.add(6, p3, 3), InvalidArgument);  // dimension mismatch
  const float nan[2] = {std::nanf(""), 0};
  CHECK_THROWS_AS(bf.add(7, nan, 2), InvalidArgument);
  CHECK_THROWS_AS(bf.remove(99), NotFound);
  CHECK_THROWS_AS(bf.knn({1, 2, 3}, 1), InvalidArgument);
  CHECK_EQ(bf.size(), 1u);  // failed operations left the index unchanged
  bf.remove(5);
  CHECK(bf.empty());
  CHECK(bf.knn(q, 3).empty());
  CHECK(bf.check_invariants().empty());
}

TEST(bruteforce_ties_break_by_smaller_id_and_swap_remove) {
  BruteForceIndex<L2> bf(1);
  // ids inserted out of order; all at distance 1 from the query 0.
  for (label_t id : {9u, 3u, 7u, 1u}) {
    const float v = (id % 2 == 0) ? -1.0f : 1.0f;
    bf.add(id, &v, 1);
  }
  const float two = 2.0f;
  bf.add(4, &two, 1);
  auto r = bf.knn({0.0f}, 3);
  REQUIRE(r.size() == 3);
  CHECK_EQ(r[0].id, 1u);
  CHECK_EQ(r[1].id, 3u);
  CHECK_EQ(r[2].id, 7u);
  bf.remove(3);  // swap-with-last moves id 4 into slot of id 3
  CHECK(bf.check_invariants().empty());
  r = bf.knn({0.0f}, 5);
  REQUIRE(r.size() == 4);
  CHECK_EQ(r[3].id, 4u);
  std::vector<float> v;
  CHECK(bf.get_vector(4, v) && v[0] == 2.0f);
  CHECK(!bf.get_vector(3, v));
}

TEST(cosine_bruteforce_rejects_zero_vectors_unchanged) {
  BruteForceIndex<Cosine> bf(3);
  const float a[3] = {1, 2, 3};
  const float z[3] = {0, 0, 0};
  bf.add(1, a, 3);
  CHECK_THROWS_AS(bf.add(2, z, 3), InvalidArgument);
  CHECK_THROWS_AS(bf.knn({0, 0, 0}, 1), InvalidArgument);
  CHECK_EQ(bf.size(), 1u);
  const auto r = bf.knn({2, 4, 6}, 1);  // same direction -> distance ~0
  REQUIRE(r.size() == 1);
  CHECK(std::abs(r[0].distance) < 1e-6f);
}
