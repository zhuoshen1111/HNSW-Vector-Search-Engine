// HNSW graph index: accuracy against the exact reference, structural invariants, determinism,
// deletion/repair/compaction state transitions and every boundary/error case of the proposal.
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "hnsw/brute_force_index.h"
#include "hnsw/hnsw_index.h"
#include "test_framework.h"
#include "test_util.h"

using namespace hnsw;

namespace {

std::vector<float> row(const std::vector<float>& data, std::size_t i, std::size_t d) {
  return std::vector<float>(data.begin() + static_cast<std::ptrdiff_t>(i * d),
                            data.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
}

/// Mean recall@k of `idx` against an exact BruteForceIndex holding the same live vectors.
template <typename Dist>
double recall_vs_exact(const Index& idx, const BruteForceIndex<Dist>& exact, const std::vector<float>& queries,
                       std::size_t nq, std::size_t d, std::size_t k, std::size_t ef) {
  double total = 0.0;
  for (std::size_t q = 0; q < nq; ++q) {
    const auto qv = row(queries, q, d);
    const auto got = idx.knn(qv, k, ef);
    const auto want = exact.knn(qv, k);
    total += testutil::overlap(got, want);
  }
  return total / static_cast<double>(nq);
}

template <typename Dist>
void check_recall_for_metric(std::uint64_t seed) {
  const std::size_t n = 3000, d = 24, nq = 100;
  const auto data = testutil::make_data(n, d, seed);
  const auto queries = testutil::make_queries(nq, d, seed);
  HnswParams p;  // proposal defaults: M = 16, efc = 200
  HNSWIndex<Dist> idx(d, p);
  BruteForceIndex<Dist> exact(d);
  for (std::size_t i = 0; i < n; ++i) {
    idx.add(i, data.data() + i * d, d);
    exact.add(i, data.data() + i * d, d);
  }
  CHECK(idx.check_invariants().empty());
  // Metric spaces reach the proposal's target at the default ef = 50. Inner product is not a
  // metric (large-norm vectors are near everything), so it needs a wider beam for the same recall.
  const std::size_t ef = Dist::kIsMetric ? 50 : 200;
  const double r = recall_vs_exact(idx, exact, queries, nq, d, 10, ef);
  CHECK(r >= 0.95);
  // keep_pruned prevents the out-degree collapse of the diversity heuristic under inner product.
  CHECK(idx.stats().avg_out_degree[0] >= static_cast<double>(p.M) / 2.0);
  for (std::size_t q = 0; q < 10; ++q) CHECK(testutil::sorted_by_order(idx.knn(row(queries, q, d), 10, 50)));
}

}  // namespace

TEST(hnsw_recall_at_default_parameters_all_metrics) {
  check_recall_for_metric<L2>(31);
  check_recall_for_metric<Cosine>(32);
  check_recall_for_metric<InnerProduct>(33);
}

TEST(hnsw_exact_when_ef_covers_the_whole_graph) {
  const std::size_t n = 400, d = 8;
  const auto data = testutil::make_data(n, d, 41);
  HNSWIndex<L2> idx(d, testutil::small_params(6, 40));
  BruteForceIndex<L2> exact(d);
  for (std::size_t i = 0; i < n; ++i) {
    idx.add(i, data.data() + i * d, d);
    exact.add(i, data.data() + i * d, d);
  }
  const auto queries = testutil::make_queries(50, d, 41);
  for (std::size_t q = 0; q < 50; ++q) {
    const auto qv = row(queries, q, d);
    CHECK(idx.knn(qv, 10, n) == exact.knn(qv, 10));  // identical ids, order and distances
  }
}

TEST(hnsw_invariants_levels_and_degrees) {
  const std::size_t n = 20000, d = 4;
  const auto data = testutil::make_data(n, d, 51, 0);
  HnswParams p;
  p.M = 16;
  p.ef_construction = 40;
  HNSWIndex<L2> idx(d, p);
  for (std::size_t i = 0; i < n; ++i) idx.add(i, data.data() + i * d, d);
  const auto errs = idx.check_invariants();
  CHECK(errs.empty());
  const IndexStats st = idx.stats();
  REQUIRE(st.nodes_per_level.size() >= 2);
  // P(level >= 1) = 1/M: expect about 1250 of 20000 nodes on layer 1.
  const double frac1 = static_cast<double>(st.nodes_per_level[1]) / static_cast<double>(n);
  CHECK(frac1 > 0.045 && frac1 < 0.08);
  CHECK(st.avg_out_degree[0] <= 32.0 && st.avg_out_degree[0] >= 4.0);
  CHECK_EQ(st.max_level, idx.max_level());
  CHECK_EQ(idx.level_of(idx.entry_id()), idx.max_level());
  CHECK_EQ(st.unreachable_live, 0u);  // a freshly built graph is fully reachable on layer 0
}

TEST(hnsw_build_is_deterministic_for_a_seed) {
  const std::size_t n = 1500, d = 12;
  const auto data = testutil::make_data(n, d, 61);
  HNSWIndex<L2> a(d, testutil::small_params());
  HNSWIndex<L2> b(d, testutil::small_params());
  HnswParams other = testutil::small_params();
  other.seed = 8;
  HNSWIndex<L2> c(d, other);
  for (std::size_t i = 0; i < n; ++i) {
    a.add(i, data.data() + i * d, d);
    b.add(i, data.data() + i * d, d);
    c.add(i, data.data() + i * d, d);
  }
  bool any_level_differs = false;
  for (label_t id = 0; id < n; ++id) {
    CHECK_EQ(a.level_of(id), b.level_of(id));
    for (int l = 0; l <= a.level_of(id); ++l) CHECK(a.neighbors_of(id, l) == b.neighbors_of(id, l));
    any_level_differs = any_level_differs || a.level_of(id) != c.level_of(id);
  }
  CHECK(any_level_differs);
}

TEST(hnsw_boundaries_empty_single_k0_and_k_above_n) {
  HNSWIndex<L2> idx(3, testutil::small_params());
  const std::vector<float> q = {0, 0, 0};
  CHECK(idx.knn(q, 5).empty());  // empty index
  CHECK(idx.check_invariants().empty());
  const float p[3] = {1, 2, 3};
  idx.add(42, p, 3);  // N = 1
  auto r = idx.knn(q, 5);
  REQUIRE(r.size() == 1);
  CHECK_EQ(r[0].id, 42u);
  CHECK_EQ(r[0].distance, 14.0f);
  CHECK(idx.knn(q, 0).empty());  // k = 0
  // k > N: every live vector comes back, in exact order, even with a tiny ef.
  const std::size_t n = 50, d = 3;
  const auto data = testutil::make_data(n, d, 71);
  HNSWIndex<L2> g(d, testutil::small_params(4, 20));
  BruteForceIndex<L2> exact(d);
  for (std::size_t i = 0; i < n; ++i) {
    g.add(i, data.data() + i * d, d);
    exact.add(i, data.data() + i * d, d);
  }
  const auto all = g.knn(q, 100, 1);
  CHECK_EQ(all.size(), n);
  CHECK(all == exact.knn(q, 100));
}

TEST(hnsw_duplicates_and_ties_break_by_id) {
  const std::size_t d = 6;
  HNSWIndex<L2> idx(d, testutil::small_params(8, 60));
  BruteForceIndex<L2> exact(d);
  const std::vector<float> same = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  const auto other = testutil::make_data(150, d, 81);
  // Interleave 150 identical vectors (ids 1000 + 2i, inserted in descending id order) with noise.
  for (std::size_t i = 0; i < 150; ++i) {
    const label_t dup_id = 1000 + 2 * (149 - i);
    idx.add(dup_id, same.data(), d);
    exact.add(dup_id, same.data(), d);
    idx.add(i, other.data() + i * d, d);
    exact.add(i, other.data() + i * d, d);
  }
  CHECK(idx.check_invariants().empty());
  const auto got = idx.knn(same, 20, 300);
  REQUIRE(got.size() == 20);
  for (std::size_t i = 0; i < 20; ++i) {
    CHECK_EQ(got[i].id, 1000u + 2 * i);  // smallest ids among the exact ties
    CHECK_EQ(got[i].distance, 0.0f);
  }
  CHECK(got == exact.knn(same, 20));
}

TEST(hnsw_rejects_bad_input_and_stays_unchanged) {
  const std::size_t d = 4;
  HNSWIndex<Cosine> idx(d, testutil::small_params());
  const auto data = testutil::make_data(100, d, 91);
  for (std::size_t i = 0; i < 100; ++i) idx.add(i, data.data() + i * d, d);
  const auto before = idx.knn(row(data, 0, d), 10, 50);
  const float zero[4] = {0, 0, 0, 0};
  const float five[5] = {1, 2, 3, 4, 5};
  const float inf[4] = {1, INFINITY, 0, 0};
  CHECK_THROWS_AS(idx.add(500, zero, d), InvalidArgument);     // zero vector under cosine
  CHECK_THROWS_AS(idx.add(501, five, 5), InvalidArgument);     // dimension mismatch
  CHECK_THROWS_AS(idx.add(502, inf, d), InvalidArgument);      // non-finite component
  CHECK_THROWS_AS(idx.add(3, data.data(), d), InvalidArgument); // duplicate live id
  CHECK_THROWS_AS(idx.remove(12345), NotFound);                // unknown id
  CHECK_THROWS_AS(idx.knn({0, 0, 0, 0}, 3), InvalidArgument);  // zero query under cosine
  CHECK_THROWS_AS(idx.knn({1, 2, 3}, 3), InvalidArgument);     // query dimension mismatch
  CHECK_EQ(idx.size(), 100u);
  CHECK_EQ(idx.slot_count(), 100u);
  CHECK(idx.check_invariants().empty());
  CHECK(idx.knn(row(data, 0, d), 10, 50) == before);
  HnswParams bad;
  bad.M = 1;
  CHECK_THROWS_AS(HNSWIndex<L2>(4, bad), InvalidArgument);
  bad.M = 8;
  bad.M0 = 4;
  CHECK_THROWS_AS(HNSWIndex<L2>(4, bad), InvalidArgument);
}

TEST(hnsw_interleaved_insert_delete_query_matches_reference) {
  for (DeletePolicy policy :
       {DeletePolicy::Repair, DeletePolicy::RepairPrune, DeletePolicy::RepairFixed, DeletePolicy::Tombstone}) {
    const std::size_t d = 10;
    HnswParams p = testutil::small_params(10, 80);
    p.delete_policy = policy;
    HNSWIndex<L2> idx(d, p);
    BruteForceIndex<L2> exact(d);
    std::mt19937_64 rng(101);
    const auto pool = testutil::make_data(4000, d, 102);
    std::vector<label_t> live;
    std::set<label_t> deleted;
    label_t next = 0;
    double recall_sum = 0.0;
    std::size_t queries = 0;
    for (int step = 0; step < 6000; ++step) {
      const int op = static_cast<int>(rng() % 10);
      if (op < 5 || live.size() < 20) {
        const std::size_t r = static_cast<std::size_t>(rng() % 4000);
        idx.add(next, pool.data() + r * d, d);
        exact.add(next, pool.data() + r * d, d);
        live.push_back(next++);
      } else if (op < 8) {
        const std::size_t j = static_cast<std::size_t>(rng() % live.size());
        const label_t id = live[j];
        live[j] = live.back();
        live.pop_back();
        idx.remove(id);
        exact.remove(id);
        deleted.insert(id);
      } else {
        const auto q = row(pool, static_cast<std::size_t>(rng() % 4000), d);
        const auto got = idx.knn(q, 10, 64);
        for (const auto& res : got) CHECK(deleted.count(res.id) == 0);  // deleted ids never appear
        CHECK(testutil::sorted_by_order(got));
        recall_sum += testutil::overlap(got, exact.knn(q, 10));
        ++queries;
      }
      if (step % 500 == 0) CHECK(idx.check_invariants().empty());
    }
    CHECK_EQ(idx.size(), live.size());
    CHECK(idx.check_invariants().empty());
    // repair-fixed is the deliberately weaker ablation (truncation strands nodes); see report E6.
    CHECK(recall_sum / static_cast<double>(queries) >= (policy == DeletePolicy::RepairFixed ? 0.85 : 0.95));
  }
}

TEST(hnsw_delete_entry_point_until_empty_then_reinsert) {
  const std::size_t n = 300, d = 5;
  const auto data = testutil::make_data(n, d, 111);
  HNSWIndex<L2> idx(d, testutil::small_params(6, 50));
  for (std::size_t i = 0; i < n; ++i) idx.add(i, data.data() + i * d, d);
  // Always delete the current entry point: forces re-election at every step.
  for (std::size_t i = 0; i < n; ++i) {
    const label_t ep = idx.entry_id();
    idx.remove(ep);
    CHECK(!idx.contains(ep));
    if (i % 25 == 0) CHECK(idx.check_invariants().empty());
    if (idx.size() > 0) {
      const auto r = idx.knn(row(data, 0, d), 5, 50);
      for (const auto& x : r) CHECK(x.id != ep);
    }
  }
  CHECK_EQ(idx.size(), 0u);
  CHECK(idx.knn(row(data, 0, d), 5).empty());  // fully deleted index
  CHECK(idx.check_invariants().empty());
  // Re-adding into a fully deleted graph must make the new node reachable and the entry point.
  idx.add(7, data.data(), d);  // id 7 was deleted earlier and may be reused
  CHECK_EQ(idx.entry_id(), 7u);
  const auto r = idx.knn(row(data, 3, d), 5);
  REQUIRE(r.size() == 1);
  CHECK_EQ(r[0].id, 7u);
  for (std::size_t i = 1; i < 40; ++i) idx.add(1000 + i, data.data() + i * d, d);
  CHECK(idx.check_invariants().empty());
  CHECK_EQ(idx.knn(row(data, 5, d), 1, 64)[0].id, 1005u);
}

TEST(hnsw_delete_hubs_keeps_recall_with_repair) {
  const std::size_t n = 3000, d = 16;
  const auto data = testutil::make_data(n, d, 121);
  HNSWIndex<L2> idx(d, testutil::small_params(12, 120));
  for (std::size_t i = 0; i < n; ++i) idx.add(i, data.data() + i * d, d);
  auto deg = idx.layer0_in_degrees();
  std::sort(deg.begin(), deg.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::set<label_t> removed;
  for (std::size_t i = 0; i < n / 10; ++i) {  // remove the 10% best-connected nodes
    idx.remove(deg[i].first);
    removed.insert(deg[i].first);
  }
  BruteForceIndex<L2> exact(d);
  for (std::size_t i = 0; i < n; ++i) {
    if (removed.count(i) == 0) exact.add(i, data.data() + i * d, d);
  }
  CHECK(idx.check_invariants().empty());
  CHECK(idx.stats().unreachable_live <= n / 100);  // degree-preserving repair keeps the graph connected
  const auto queries = testutil::make_queries(100, d, 121);
  CHECK(recall_vs_exact(idx, exact, queries, 100, d, 10, 64) >= 0.95);
}

TEST(hnsw_repair_removes_more_tombstone_edges_than_marking) {
  const std::size_t n = 2000, d = 8;
  const auto data = testutil::make_data(n, d, 131);
  HnswParams pt = testutil::small_params();
  pt.delete_policy = DeletePolicy::Tombstone;
  HnswParams pr = testutil::small_params();
  pr.delete_policy = DeletePolicy::Repair;
  HNSWIndex<L2> tomb(d, pt);
  HNSWIndex<L2> rep(d, pr);
  for (std::size_t i = 0; i < n; ++i) {
    tomb.add(i, data.data() + i * d, d);
    rep.add(i, data.data() + i * d, d);
  }
  for (label_t id = 0; id < n; id += 3) {
    tomb.remove(id);
    rep.remove(id);
  }
  const IndexStats st = tomb.stats();
  const IndexStats sr = rep.stats();
  CHECK_EQ(st.tombstones, sr.tombstones);
  CHECK(st.edges_to_tombstones > 0);
  CHECK(sr.edges_to_tombstones * 4 < st.edges_to_tombstones);
  CHECK(tomb.check_invariants().empty());
  CHECK(rep.check_invariants().empty());
}

TEST(hnsw_compact_drops_tombstones_and_preserves_answers) {
  const std::size_t n = 2000, d = 8;
  const auto data = testutil::make_data(n, d, 141);
  HNSWIndex<L2> idx(d, testutil::small_params());
  HNSWIndex<L2> fresh(d, testutil::small_params());  // built directly from the survivors
  BruteForceIndex<L2> exact(d);
  for (std::size_t i = 0; i < n; ++i) idx.add(i, data.data() + i * d, d);
  for (std::size_t i = 0; i < n; ++i) {
    if (i % 2 == 0) {
      idx.remove(i);
    } else {
      exact.add(i, data.data() + i * d, d);
      fresh.add(i, data.data() + i * d, d);
    }
  }
  CHECK_EQ(idx.tombstone_count(), n / 2);
  idx.compact();
  CHECK_EQ(idx.tombstone_count(), 0u);
  CHECK_EQ(idx.slot_count(), n / 2);
  CHECK_EQ(idx.size(), n / 2);
  CHECK(idx.check_invariants().empty());
  const auto queries = testutil::make_queries(100, d, 141);
  // Compaction re-inserts the survivors in slot order with the same seed, so the result is the
  // very same graph as building from the survivors directly.
  for (std::size_t q = 0; q < 100; ++q) CHECK(idx.knn(row(queries, q, d), 10, 64) == fresh.knn(row(queries, q, d), 10, 64));
  CHECK(recall_vs_exact(idx, exact, queries, 100, d, 10, 200) >= 0.99);
  std::vector<float> v;
  CHECK(idx.get_vector(1, v) && v == row(data, 1, d));
  CHECK(!idx.get_vector(0, v));
  idx.add(0, data.data(), d);  // deleted ids are reusable after compaction too
  CHECK(idx.contains(0));
}

TEST(nsw_single_layer_and_simple_selection_variants) {
  const std::size_t n = 3000, d = 16, nq = 100;
  // Uniform data: every variant (NSW / simple selection) builds a navigable graph.
  const auto uni = testutil::make_data(n, d, 151, 0);
  const auto uq = testutil::make_queries(nq, d, 151, 0);
  BruteForceIndex<L2> exact(d);
  for (std::size_t i = 0; i < n; ++i) exact.add(i, uni.data() + i * d, d);
  HnswParams nsw = testutil::small_params(12, 100);
  nsw.single_layer = true;
  HnswParams simple = testutil::small_params(12, 100);
  simple.selection = NeighborSelection::Simple;
  HNSWIndex<L2> a(d, nsw);
  HNSWIndex<L2> b(d, simple);
  for (std::size_t i = 0; i < n; ++i) {
    a.add(i, uni.data() + i * d, d);
    b.add(i, uni.data() + i * d, d);
  }
  CHECK_EQ(a.max_level(), 0);
  CHECK_EQ(a.stats().nodes_per_level.size(), 1u);
  CHECK_EQ(a.stats().kind, std::string("nsw"));
  CHECK(a.check_invariants().empty());
  CHECK(b.check_invariants().empty());
  CHECK(recall_vs_exact(a, exact, uq, nq, d, 10, 64) >= 0.95);
  CHECK(recall_vs_exact(b, exact, uq, nq, d, 10, 64) >= 0.95);
}

TEST(heuristic_selection_beats_simple_on_clustered_data) {
  // With well-separated clusters, keeping only the M closest candidates links each cluster mostly
  // to itself and the beam gets stuck; the diversity heuristic keeps long-range links (paper 4.1).
  const std::size_t n = 3000, d = 16, nq = 100;
  double simple_sum = 0.0;
  for (std::uint64_t seed : {151u, 152u, 153u}) {
    const auto data = testutil::make_data(n, d, seed, 16);
    const auto queries = testutil::make_queries(nq, d, seed, 16);
    BruteForceIndex<L2> exact(d);
    HnswParams simple = testutil::small_params(12, 100);
    simple.selection = NeighborSelection::Simple;
    HNSWIndex<L2> s(d, simple);
    HNSWIndex<L2> h(d, testutil::small_params(12, 100));
    for (std::size_t i = 0; i < n; ++i) {
      exact.add(i, data.data() + i * d, d);
      s.add(i, data.data() + i * d, d);
      h.add(i, data.data() + i * d, d);
    }
    CHECK(recall_vs_exact(h, exact, queries, nq, d, 10, 64) >= 0.99);
    simple_sum += recall_vs_exact(s, exact, queries, nq, d, 10, 64);
  }
  CHECK(simple_sum / 3.0 < 0.95);
}

TEST(hnsw_search_stats_count_work) {
  const std::size_t n = 2000, d = 8;
  const auto data = testutil::make_data(n, d, 161);
  HNSWIndex<L2> idx(d, testutil::small_params());
  for (std::size_t i = 0; i < n; ++i) idx.add(i, data.data() + i * d, d);
  SearchParams small;
  small.ef = 10;
  SearchParams large;
  large.ef = 200;
  SearchStats s1, s2;
  const auto q = row(data, 5, d);
  idx.search(q.data(), d, 10, small, &s1);
  idx.search(q.data(), d, 10, large, &s2);
  CHECK(s1.distance_computations > 0);
  CHECK(s2.distance_computations > s1.distance_computations);  // larger beam, more work
  CHECK(s2.distance_computations < n);                         // still far below a full scan
  CHECK(s1.expanded <= s1.visited);
}
