// hnsw_bench: controlled experiments. Every sub-command appends one CSV row per measured setting.
//
//   sweep   build an index once per repetition, then sweep ef (and k): recall, latency, work, memory
//   delete  delete a growing fraction of the vectors (random or hubs-first) under a delete policy
//   churn   steady-state replace workload: each round deletes and inserts a fraction of the data
//   topk    exact top-k selection microbenchmark: quickselect vs bounded heap vs full sort
//   kernel  distance kernel microbenchmark: scalar reference vs SIMD
//
// Timing: std::chrono::steady_clock wall time, single thread. Each timed setting is run once as
// warm-up (which also produces the recall and work counters) and then `--query-reps` more times;
// the CSV holds the median over repetitions and the standard deviation of the per-rep means.
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>

#include "bench_common.h"
#include "hnsw/brute_force_index.h"
#include "hnsw/hnsw_index.h"
#include "hnsw/index_factory.h"
#include "hnsw/io.h"
#include "hnsw/timer.h"
#include "hnsw/top_k.h"

using bench::fmt;
using hnsw::label_t;

namespace {

struct Dataset {
  std::string name;
  hnsw::Matrix base;
  hnsw::Matrix queries;
  std::vector<std::vector<label_t>> gt;  // exact neighbours of the full base set (may be empty)
};

Dataset load_dataset(const bench::ArgMap& a, hnsw::Metric metric, std::size_t need_k) {
  Dataset d;
  d.base = hnsw::read_fvecs(a.required("base"), a.size("n", hnsw::kNoLimit));
  d.queries = hnsw::read_fvecs(a.required("query"), a.size("nq", hnsw::kNoLimit));
  d.name = a.str("dataset", a.required("base"));
  if (d.base.cols != d.queries.cols) throw hnsw::InvalidArgument("base and query dimensions differ");
  bool have_gt = false;
  if (a.has("gt")) {
    const auto rows = hnsw::read_ivecs(a.str("gt", ""), d.queries.rows);
    // A ground-truth file is valid only for the full base set it was computed on.
    bool ok = rows.size() >= d.queries.rows && !rows.empty() && rows[0].size() >= need_k;
    for (const auto& r : rows) {
      for (std::int32_t id : r) ok = ok && id >= 0 && static_cast<std::size_t>(id) < d.base.rows;
    }
    if (ok) {
      for (const auto& r : rows) d.gt.emplace_back(r.begin(), r.end());
      have_gt = true;
    } else {
      std::cerr << "note: ground truth does not match this subset; recomputing exactly\n";
    }
  }
  if (!have_gt) {
    const hnsw::Timer t;
    d.gt = bench::exact_neighbors(metric, d.base, {}, d.queries, std::max<std::size_t>(need_k, 1),
                                  static_cast<unsigned>(a.size("threads", bench::default_threads())));
    std::cerr << "computed exact ground truth in " << fmt(t.seconds(), 3) << " s\n";
  }
  return d;
}

hnsw::IndexConfig config_from(const bench::ArgMap& a, hnsw::Metric metric, std::size_t dim) {
  hnsw::IndexConfig c;
  c.metric = metric;
  c.dim = dim;
  const std::string kind = a.str("index", "hnsw");
  c.kind = kind == "bruteforce" ? hnsw::IndexKind::BruteForce : hnsw::IndexKind::HNSW;
  if (kind != "bruteforce" && kind != "hnsw" && kind != "nsw") throw hnsw::InvalidArgument("bad --index");
  c.hnsw.single_layer = kind == "nsw";
  c.hnsw.M = a.size("M", 16);
  c.hnsw.M0 = a.size("M0", 0);
  c.hnsw.ef_construction = a.size("efc", 200);
  c.hnsw.seed = a.u64("seed", 42);
  c.hnsw.selection = hnsw::parse_selection(a.str("select", "heuristic"));
  return c;
}

struct QueryMeasurement {
  double recall = 0.0;
  double dist_per_query = 0.0;
  double visited_per_query = 0.0;
  double lat_mean_us = 0.0;      // median over reps of the per-rep mean latency
  double lat_mean_us_std = 0.0;  // std of the per-rep means
  double lat_p50_us = 0.0;
  double lat_p99_us = 0.0;
  double qps = 0.0;
  std::size_t reps = 0;
};

QueryMeasurement measure_queries(const hnsw::Index& idx, const hnsw::Matrix& queries,
                                 const std::vector<std::vector<label_t>>& truth, std::size_t k, std::size_t ef,
                                 std::size_t reps) {
  QueryMeasurement m;
  hnsw::SearchParams p;
  p.ef = ef;
  // Warm-up pass: correctness and work counters.
  std::vector<std::vector<hnsw::SearchResult>> results(queries.rows);
  hnsw::SearchStats work;
  for (std::size_t q = 0; q < queries.rows; ++q) results[q] = idx.search(queries.row(q), queries.cols, k, p, &work);
  m.recall = bench::recall_at_k(results, truth, k);
  m.dist_per_query = static_cast<double>(work.distance_computations) / static_cast<double>(queries.rows);
  m.visited_per_query = static_cast<double>(work.visited) / static_cast<double>(queries.rows);
  // Timed passes.
  std::vector<double> means, p50s, p99s, qpss;
  std::vector<double> lat(queries.rows);
  volatile std::size_t sink = 0;
  for (std::size_t r = 0; r < reps; ++r) {
    const hnsw::Timer total;
    for (std::size_t q = 0; q < queries.rows; ++q) {
      const hnsw::Timer t;
      const auto res = idx.search(queries.row(q), queries.cols, k, p, nullptr);
      lat[q] = t.micros();
      sink = sink + res.size();
    }
    const double secs = total.seconds();
    const hnsw::Summary s = hnsw::summarize(lat);
    means.push_back(s.mean);
    p50s.push_back(s.median);
    p99s.push_back(s.p99);
    qpss.push_back(static_cast<double>(queries.rows) / secs);
  }
  const hnsw::Summary sm = hnsw::summarize(means);
  m.lat_mean_us = sm.median;
  m.lat_mean_us_std = sm.stddev;
  m.lat_p50_us = hnsw::summarize(p50s).median;
  m.lat_p99_us = hnsw::summarize(p99s).median;
  m.qps = hnsw::summarize(qpss).median;
  m.reps = reps;
  return m;
}

/// Truth restricted to the first k ids of each row.
std::vector<std::vector<label_t>> truncate(const std::vector<std::vector<label_t>>& gt, std::size_t k) {
  std::vector<std::vector<label_t>> out;
  out.reserve(gt.size());
  for (const auto& r : gt) out.emplace_back(r.begin(), r.begin() + static_cast<std::ptrdiff_t>(std::min(k, r.size())));
  return out;
}

std::unique_ptr<hnsw::Index> build(const hnsw::IndexConfig& cfg, const hnsw::Matrix& base, std::size_t rows,
                                   double* seconds) {
  std::unique_ptr<hnsw::Index> idx = hnsw::make_index(cfg);
  idx->reserve(rows);
  const hnsw::Timer t;
  for (std::size_t i = 0; i < rows; ++i) idx->add(i, base.row(i), base.cols);
  if (seconds) *seconds = t.seconds();
  return idx;
}

// ------------------------------------------------------------------------------------------
int cmd_sweep(const bench::ArgMap& a) {
  a.allow({"base", "query", "gt", "n", "nq", "dataset", "metric", "index", "M", "M0", "efc", "seed", "select", "efs",
           "ks", "build-reps", "query-reps", "out", "tag", "threads"});
  const hnsw::Metric metric = hnsw::parse_metric(a.str("metric", "l2"));
  const std::vector<std::size_t> ks = a.sizes("ks", {10});
  const Dataset ds = load_dataset(a, metric, *std::max_element(ks.begin(), ks.end()));
  const hnsw::IndexConfig cfg = config_from(a, metric, ds.base.cols);
  const std::size_t build_reps = std::max<std::size_t>(1, a.size("build-reps", 1));
  const std::size_t query_reps = std::max<std::size_t>(1, a.size("query-reps", 5));
  std::vector<std::size_t> efs = a.sizes("efs", {10, 20, 40, 80, 160, 320});
  if (cfg.kind == hnsw::IndexKind::BruteForce) efs = {0};

  const std::size_t rss_before = bench::current_rss_bytes();
  std::vector<double> build_times;
  std::unique_ptr<hnsw::Index> idx;
  for (std::size_t r = 0; r < build_reps; ++r) {
    idx.reset();
    double secs = 0.0;
    idx = build(cfg, ds.base, ds.base.rows, &secs);
    build_times.push_back(secs);
    std::cerr << "  build " << (r + 1) << "/" << build_reps << ": " << fmt(secs, 4) << " s\n";
  }
  const std::size_t rss_after = bench::current_rss_bytes();
  const hnsw::IndexStats st = idx->stats();
  const hnsw::Summary bt = hnsw::summarize(build_times);

  const std::string index_name = cfg.kind == hnsw::IndexKind::BruteForce ? "bruteforce" : st.kind;
  bench::CsvWriter csv(a.required("out"),
                       {"tag", "dataset", "metric", "n", "dim", "nq", "index", "M", "efc", "select", "seed",
                        "build_reps", "build_s", "build_s_std", "ef", "k", "recall", "query_reps", "lat_mean_us",
                        "lat_mean_us_std", "lat_p50_us", "lat_p99_us", "qps", "dist_per_query", "visited_per_query",
                        "index_bytes", "bytes_per_vector", "rss_delta_bytes", "peak_rss_bytes", "max_level",
                        "avg_deg0", "simd", "machine"});
  for (const std::size_t k : ks) {
    const auto truth = truncate(ds.gt, k);
    for (const std::size_t ef : efs) {
      const QueryMeasurement m = measure_queries(*idx, ds.queries, truth, k, ef, query_reps);
      std::cerr << "  " << index_name << " k=" << k << " ef=" << ef << ": recall " << fmt(m.recall, 4) << ", "
                << fmt(m.lat_mean_us, 4) << " us/query\n";
      csv.row({a.str("tag", ""), ds.name, hnsw::metric_name(metric), std::to_string(ds.base.rows),
               std::to_string(ds.base.cols), std::to_string(ds.queries.rows), index_name,
               std::to_string(cfg.hnsw.M), std::to_string(cfg.hnsw.ef_construction),
               hnsw::selection_name(cfg.hnsw.selection), std::to_string(cfg.hnsw.seed), std::to_string(build_reps),
               fmt(bt.median), fmt(bt.stddev), std::to_string(ef), std::to_string(k), fmt(m.recall),
               std::to_string(m.reps), fmt(m.lat_mean_us), fmt(m.lat_mean_us_std), fmt(m.lat_p50_us),
               fmt(m.lat_p99_us), fmt(m.qps), fmt(m.dist_per_query), fmt(m.visited_per_query),
               std::to_string(st.total_bytes()), fmt(st.bytes_per_live_vector()),
               std::to_string(rss_after > rss_before ? rss_after - rss_before : 0),
               std::to_string(bench::peak_rss_bytes()), std::to_string(st.max_level),
               fmt(st.avg_out_degree.empty() ? 0.0 : st.avg_out_degree[0]), hnsw::kernels::simd_name(),
               "\"" + bench::machine_description() + "\""});
    }
  }
  return 0;
}

// ------------------------------------------------------------------------------------------
template <typename Dist>
bool hub_order_for(const hnsw::Index& idx, std::vector<label_t>& order) {
  const auto* h = dynamic_cast<const hnsw::HNSWIndex<Dist>*>(&idx);
  if (h == nullptr) return false;
  auto deg = h->layer0_in_degrees();
  std::sort(deg.begin(), deg.end(), [](const auto& x, const auto& y) {
    return x.second != y.second ? x.second > y.second : x.first < y.first;
  });
  order.clear();
  for (const auto& p : deg) order.push_back(p.first);
  return true;
}

std::vector<label_t> deletion_order(const hnsw::Index& idx, const std::string& how, std::uint64_t seed) {
  std::vector<label_t> order = idx.labels();
  if (how == "hubs") {
    if (hub_order_for<hnsw::L2>(idx, order) || hub_order_for<hnsw::InnerProduct>(idx, order) ||
        hub_order_for<hnsw::Cosine>(idx, order)) {
      return order;
    }
    throw hnsw::InvalidArgument("--order hubs needs an HNSW index");
  }
  if (how != "random") throw hnsw::InvalidArgument("--order must be random or hubs");
  bench::Rng rng(seed);
  for (std::size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[static_cast<std::size_t>(rng.below(i))]);
  return order;
}

/// Exact neighbours over the currently live ids (ids are row numbers of `base`).
std::vector<std::vector<label_t>> live_truth(const hnsw::Index& idx, hnsw::Metric metric, const hnsw::Matrix& base,
                                             const std::vector<std::size_t>& row_of, const hnsw::Matrix& queries,
                                             std::size_t k, unsigned threads) {
  const std::vector<label_t> live = idx.labels();
  hnsw::Matrix sub;
  sub.cols = base.cols;
  sub.rows = live.size();
  sub.data.resize(sub.rows * sub.cols);
  for (std::size_t i = 0; i < live.size(); ++i) {
    const float* src = base.row(row_of.empty() ? static_cast<std::size_t>(live[i]) : row_of[live[i]]);
    std::copy(src, src + base.cols, sub.row(i));
  }
  return bench::exact_neighbors(metric, sub, live, queries, k, threads);
}

int cmd_delete(const bench::ArgMap& a) {
  a.allow({"base", "query", "gt", "n", "nq", "dataset", "metric", "index", "M", "M0", "efc", "seed", "select", "policy",
           "fractions", "efs", "k", "query-reps", "order", "out", "tag", "threads"});
  const hnsw::Metric metric = hnsw::parse_metric(a.str("metric", "l2"));
  const std::size_t k = a.size("k", 10);
  Dataset ds;
  ds.base = hnsw::read_fvecs(a.required("base"), a.size("n", hnsw::kNoLimit));
  ds.queries = hnsw::read_fvecs(a.required("query"), a.size("nq", hnsw::kNoLimit));
  ds.name = a.str("dataset", a.required("base"));
  const std::string policy = a.str("policy", "repair");
  hnsw::IndexConfig cfg = config_from(a, metric, ds.base.cols);
  if (policy == "tombstone" || policy == "rebuild") {
    cfg.hnsw.delete_policy = hnsw::DeletePolicy::Tombstone;
  } else {
    cfg.hnsw.delete_policy = hnsw::parse_delete_policy(policy);  // repair | repair-prune
  }
  const std::vector<double> fractions = a.reals("fractions", {0.0, 0.1, 0.2, 0.3, 0.4, 0.5});
  const std::vector<std::size_t> efs = a.sizes("efs", {50});
  const std::size_t reps = std::max<std::size_t>(1, a.size("query-reps", 5));
  const unsigned threads = static_cast<unsigned>(a.size("threads", bench::default_threads()));
  const std::string order_kind = a.str("order", "random");

  double build_s = 0.0;
  std::unique_ptr<hnsw::Index> idx = build(cfg, ds.base, ds.base.rows, &build_s);
  const std::vector<label_t> order = deletion_order(*idx, order_kind, a.u64("seed", 42) + 7);
  std::cerr << "built in " << fmt(build_s, 4) << " s; deleting in " << order_kind << " order with " << policy << "\n";

  bench::CsvWriter csv(a.required("out"),
                       {"tag", "dataset", "metric", "n", "dim", "nq", "M", "efc", "policy", "order", "fraction",
                        "deleted", "live", "delete_us_per_op", "rebuild_s", "ef", "k", "recall", "query_reps",
                        "lat_mean_us", "lat_mean_us_std", "lat_p99_us", "dist_per_query", "tombstones", "edges",
                        "edges_to_tombstones", "unreachable_live", "index_bytes", "avg_deg0"});
  std::size_t deleted = 0;
  for (const double f : fractions) {
    const std::size_t target = std::min(order.size(), static_cast<std::size_t>(f * static_cast<double>(ds.base.rows) + 0.5));
    const std::size_t batch = target > deleted ? target - deleted : 0;
    const hnsw::Timer t;
    for (; deleted < target; ++deleted) idx->remove(order[deleted]);
    const double del_us = batch == 0 ? 0.0 : t.micros() / static_cast<double>(batch);
    double rebuild_s = 0.0;
    if (policy == "rebuild") {
      const hnsw::Timer tr;
      idx->compact();
      rebuild_s = tr.seconds();
    }
    const auto truth = live_truth(*idx, metric, ds.base, {}, ds.queries, k, threads);
    const hnsw::IndexStats st = idx->stats();
    for (const std::size_t ef : efs) {
      const QueryMeasurement m = measure_queries(*idx, ds.queries, truth, k, ef, reps);
      std::cerr << "  f=" << f << " ef=" << ef << ": recall " << fmt(m.recall, 4) << ", " << fmt(m.lat_mean_us, 4)
                << " us, delete " << fmt(del_us, 4) << " us/op\n";
      csv.row({a.str("tag", ""), ds.name, hnsw::metric_name(metric), std::to_string(ds.base.rows),
               std::to_string(ds.base.cols), std::to_string(ds.queries.rows), std::to_string(cfg.hnsw.M),
               std::to_string(cfg.hnsw.ef_construction), policy, order_kind, fmt(f), std::to_string(deleted),
               std::to_string(idx->size()), fmt(del_us), fmt(rebuild_s), std::to_string(ef), std::to_string(k),
               fmt(m.recall), std::to_string(m.reps), fmt(m.lat_mean_us), fmt(m.lat_mean_us_std), fmt(m.lat_p99_us),
               fmt(m.dist_per_query), std::to_string(st.tombstones), std::to_string(st.edges),
               std::to_string(st.edges_to_tombstones), std::to_string(st.unreachable_live),
               std::to_string(st.total_bytes()), fmt(st.avg_out_degree.empty() ? 0.0 : st.avg_out_degree[0])});
    }
  }
  return 0;
}

// ------------------------------------------------------------------------------------------
int cmd_churn(const bench::ArgMap& a) {
  a.allow({"base", "query", "n", "nq", "dataset", "metric", "M", "M0", "efc", "seed", "select", "policy", "initial",
           "rounds", "rate", "rebuild-every", "ef", "k", "query-reps", "out", "tag", "threads"});
  const hnsw::Metric metric = hnsw::parse_metric(a.str("metric", "l2"));
  const hnsw::Matrix base = hnsw::read_fvecs(a.required("base"), a.size("n", hnsw::kNoLimit));
  const hnsw::Matrix queries = hnsw::read_fvecs(a.required("query"), a.size("nq", hnsw::kNoLimit));
  const std::string policy = a.str("policy", "repair");
  hnsw::IndexConfig cfg = config_from(a, metric, base.cols);
  cfg.hnsw.delete_policy = (policy == "tombstone" || policy == "rebuild") ? hnsw::DeletePolicy::Tombstone
                                                                         : hnsw::parse_delete_policy(policy);
  const std::size_t rebuild_every = std::max<std::size_t>(1, a.size("rebuild-every", 5));
  const std::size_t n0 = static_cast<std::size_t>(a.real("initial", 0.5) * static_cast<double>(base.rows));
  const std::size_t rounds = a.size("rounds", 10);
  const double rate = a.real("rate", 0.05);
  const std::size_t per_round = std::max<std::size_t>(1, static_cast<std::size_t>(rate * static_cast<double>(n0)));
  const std::size_t k = a.size("k", 10);
  const std::size_t ef = a.size("ef", 50);
  const std::size_t reps = std::max<std::size_t>(1, a.size("query-reps", 3));
  const unsigned threads = static_cast<unsigned>(a.size("threads", bench::default_threads()));

  std::unique_ptr<hnsw::Index> idx = hnsw::make_index(cfg);
  std::vector<std::size_t> row_of;  // label -> base row
  label_t next_label = 0;
  std::size_t next_row = 0;
  auto insert_one = [&]() {
    const std::size_t row = next_row;
    next_row = (next_row + 1) % base.rows;
    idx->add(next_label, base.row(row), base.cols);
    row_of.push_back(row);
    ++next_label;
  };
  for (std::size_t i = 0; i < n0; ++i) insert_one();
  bench::Rng rng(a.u64("seed", 42) + 99);
  bench::CsvWriter csv(a.required("out"),
                       {"tag", "dataset", "metric", "n0", "dim", "policy", "rate", "round", "live", "tombstones",
                        "update_us_per_op", "rebuild_s", "ef", "k", "recall", "lat_mean_us", "lat_mean_us_std",
                        "dist_per_query", "edges_to_tombstones", "unreachable_live", "index_bytes"});
  for (std::size_t round = 0; round <= rounds; ++round) {
    double upd_us = 0.0;
    double rebuild_s = 0.0;
    if (round > 0) {
      std::vector<label_t> live = idx->labels();
      for (std::size_t i = 0; i < per_round && i < live.size(); ++i) {
        const std::size_t j = i + static_cast<std::size_t>(rng.below(live.size() - i));
        std::swap(live[i], live[j]);
      }
      const hnsw::Timer t;
      for (std::size_t i = 0; i < per_round && i < live.size(); ++i) idx->remove(live[i]);
      for (std::size_t i = 0; i < per_round; ++i) insert_one();
      upd_us = t.micros() / static_cast<double>(2 * per_round);
      if (policy == "rebuild" && round % rebuild_every == 0) {
        const hnsw::Timer tr;
        idx->compact();
        rebuild_s = tr.seconds();
      }
    }
    const auto truth = live_truth(*idx, metric, base, row_of, queries, k, threads);
    const QueryMeasurement m = measure_queries(*idx, queries, truth, k, ef, reps);
    const hnsw::IndexStats st = idx->stats();
    std::cerr << "  round " << round << ": recall " << fmt(m.recall, 4) << ", " << fmt(m.lat_mean_us, 4)
              << " us, tombstones " << st.tombstones << "\n";
    csv.row({a.str("tag", ""), a.str("dataset", a.required("base")), hnsw::metric_name(metric), std::to_string(n0),
             std::to_string(base.cols), policy, fmt(rate), std::to_string(round), std::to_string(idx->size()),
             std::to_string(st.tombstones), fmt(upd_us), fmt(rebuild_s), std::to_string(ef), std::to_string(k),
             fmt(m.recall), fmt(m.lat_mean_us), fmt(m.lat_mean_us_std), fmt(m.dist_per_query),
             std::to_string(st.edges_to_tombstones), std::to_string(st.unreachable_live),
             std::to_string(st.total_bytes())});
  }
  return 0;
}

// ------------------------------------------------------------------------------------------
int cmd_topk(const bench::ArgMap& a) {
  a.allow({"ns", "ks", "reps", "seed", "out"});
  const std::vector<std::size_t> ns = a.sizes("ns", {10000, 100000, 1000000});
  const std::vector<std::size_t> ks = a.sizes("ks", {1, 10, 100});
  const std::size_t reps = std::max<std::size_t>(1, a.size("reps", 5));
  bench::CsvWriter csv(a.required("out"), {"n", "k", "method", "reps", "median_us", "std_us", "ns_per_element"});
  bench::Rng rng(a.u64("seed", 3));
  for (const std::size_t n : ns) {
    std::vector<hnsw::SearchResult> input(n);
    for (std::size_t i = 0; i < n; ++i) input[i] = {i, static_cast<float>(rng.uniform())};
    for (const std::size_t k : ks) {
      std::vector<hnsw::SearchResult> reference;
      for (const char* method : {"quickselect", "heap", "fullsort"}) {
        std::vector<double> times;
        std::vector<hnsw::SearchResult> out;
        for (std::size_t r = 0; r < reps + 1; ++r) {
          std::vector<hnsw::SearchResult> work = input;  // copy outside the timed region
          const hnsw::Timer t;
          if (std::string(method) == "quickselect") {
            hnsw::select_smallest(work.begin(), work.size(), k, hnsw::ResultOrder{});
            work.resize(std::min(k, work.size()));
            std::sort(work.begin(), work.end(), hnsw::ResultOrder{});
            out = std::move(work);
          } else if (std::string(method) == "heap") {
            hnsw::BoundedMaxHeap<hnsw::SearchResult, hnsw::ResultOrder> h(k);
            for (const auto& x : work) h.push(x);
            out = h.take_sorted();
          } else {
            std::sort(work.begin(), work.end(), hnsw::ResultOrder{});
            work.resize(std::min(k, work.size()));
            out = std::move(work);
          }
          if (r > 0) times.push_back(t.micros());  // first run is warm-up
        }
        if (reference.empty()) reference = out;
        if (out != reference) throw hnsw::Error(std::string("top-k mismatch for ") + method);
        const hnsw::Summary s = hnsw::summarize(times);
        std::cerr << "  n=" << n << " k=" << k << " " << method << ": " << fmt(s.median, 4) << " us\n";
        csv.row({std::to_string(n), std::to_string(k), method, std::to_string(reps), fmt(s.median), fmt(s.stddev),
                 fmt(s.median * 1000.0 / static_cast<double>(n))});
      }
    }
  }
  return 0;
}

// ------------------------------------------------------------------------------------------
int cmd_kernel(const bench::ArgMap& a) {
  a.allow({"dims", "vectors", "reps", "seed", "out"});
  const std::vector<std::size_t> dims = a.sizes("dims", {32, 128, 960});
  const std::size_t nvec = a.size("vectors", 4096);
  const std::size_t reps = std::max<std::size_t>(1, a.size("reps", 5));
  bench::CsvWriter csv(a.required("out"),
                       {"dim", "kernel", "impl", "reps", "ns_per_distance", "std_ns", "max_rel_error", "simd"});
  bench::Rng rng(a.u64("seed", 5));
  for (const std::size_t d : dims) {
    std::vector<float> data(nvec * d);
    for (float& x : data) x = static_cast<float>(rng.uniform() * 2.0 - 1.0);
    const std::size_t pairs = nvec - 1;
    for (const char* kernel : {"l2", "dot"}) {
      double max_rel = 0.0;
      for (std::size_t i = 0; i < pairs; ++i) {
        const float* x = data.data() + i * d;
        const float* y = x + d;
        const double ref = std::string(kernel) == "l2" ? hnsw::kernels::l2_sqr_scalar(x, y, d)
                                                       : hnsw::kernels::dot_scalar(x, y, d);
        const double fast = std::string(kernel) == "l2" ? hnsw::kernels::l2_sqr_simd(x, y, d)
                                                        : hnsw::kernels::dot_simd(x, y, d);
        max_rel = std::max(max_rel, std::abs(fast - ref) / std::max(1e-6, std::abs(ref)));
      }
      for (const char* impl : {"scalar", "simd"}) {
        const bool simd = std::string(impl) == "simd";
        std::vector<double> times;
        volatile float sink = 0.0f;
        const std::size_t sweeps = std::max<std::size_t>(1, 20000000 / (pairs * d) + 1);
        for (std::size_t r = 0; r < reps + 1; ++r) {
          float acc = 0.0f;
          const hnsw::Timer t;
          for (std::size_t s = 0; s < sweeps; ++s) {
            for (std::size_t i = 0; i < pairs; ++i) {
              const float* x = data.data() + i * d;
              const float* y = x + d;
              if (std::string(kernel) == "l2") {
                acc += simd ? hnsw::kernels::l2_sqr_simd(x, y, d) : hnsw::kernels::l2_sqr_scalar(x, y, d);
              } else {
                acc += simd ? hnsw::kernels::dot_simd(x, y, d) : hnsw::kernels::dot_scalar(x, y, d);
              }
            }
          }
          const double ns = t.micros() * 1000.0 / static_cast<double>(sweeps * pairs);
          sink = sink + acc;
          if (r > 0) times.push_back(ns);
        }
        const hnsw::Summary s = hnsw::summarize(times);
        std::cerr << "  d=" << d << " " << kernel << " " << impl << ": " << fmt(s.median, 4) << " ns\n";
        csv.row({std::to_string(d), kernel, impl, std::to_string(reps), fmt(s.median), fmt(s.stddev), fmt(max_rel),
                 hnsw::kernels::simd_name()});
      }
    }
  }
  return 0;
}

void usage() {
  std::cerr << "usage: hnsw_bench <sweep|delete|churn|topk|kernel> --out results.csv [options]\n"
               "  sweep  --base B.fvecs --query Q.fvecs [--gt G.ivecs] [--n N] [--nq Q] [--metric l2]\n"
               "         [--index hnsw|nsw|bruteforce] [--M 16] [--efc 200] [--select heuristic|simple]\n"
               "         [--efs 10,20,40] [--ks 10] [--build-reps 1] [--query-reps 5] [--tag T]\n"
               "  delete --base B --query Q [--policy tombstone|repair|repair-prune|repair-fixed|rebuild]\n"
               "         [--order random|hubs]\n"
               "         [--fractions 0,0.1,0.2] [--efs 50] [--k 10]\n"
               "  churn  --base B --query Q [--policy repair|repair-prune|tombstone|rebuild] [--initial 0.5] [--rounds 10]\n"
               "         [--rate 0.05] [--rebuild-every 5] [--ef 50] [--k 10]\n"
               "  topk   [--ns 10000,100000] [--ks 1,10,100] [--reps 5]\n"
               "  kernel [--dims 32,128,960] [--reps 5]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  try {
    const std::string sub = argv[1];
    const bench::ArgMap args(argc, argv, 2);
    if (sub == "sweep") return cmd_sweep(args);
    if (sub == "delete") return cmd_delete(args);
    if (sub == "churn") return cmd_churn(args);
    if (sub == "topk") return cmd_topk(args);
    if (sub == "kernel") return cmd_kernel(args);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
