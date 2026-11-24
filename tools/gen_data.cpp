// hnsw_gen: seeded synthetic dataset generator with exact ground truth.
//
//   hnsw_gen --out data/syn --type clustered --n 100000 --nq 1000 --dim 128 [--clusters 64]
//            [--latent 16] [--spread 0.1] [--noise 0.01] [--seed 1] [--metric l2] [--gt-k 100]
//            [--threads T] [--csv]
//
// Types:
//   uniform    x ~ U[0,1]^d. Intrinsic dimension = d, the hardest case for graph search.
//   clustered  latent z = c_j + spread * N(0, I_r) with c_j ~ U[0,1]^r for a random cluster j,
//              then x = A z + noise * N(0, I_d) with a fixed random A (d x r, N(0, 1/d) entries).
//              Mimics embedding data: low intrinsic dimension r, cluster structure, full ambient d.
// Writes <out>_base.fvecs, <out>_query.fvecs, <out>_gt.ivecs (exact top gt-k ids under --metric)
// and <out>_meta.txt with every parameter; --csv also writes small CSV copies.
#include <fstream>
#include <iostream>

#include "bench_common.h"
#include "hnsw/io.h"
#include "hnsw/timer.h"

namespace {

struct GenParams {
  std::string type;
  std::size_t dim, clusters, latent;
  double spread, noise;
};

class Generator {
 public:
  Generator(const GenParams& p, std::uint64_t seed) : p_(p), rng_(seed) {
    if (p_.type == "clustered") {
      r_ = std::min(p_.latent, p_.dim);
      centers_.resize(p_.clusters * r_);
      for (double& c : centers_) c = rng_.uniform();
      proj_.resize(p_.dim * r_);
      const double scale = 1.0 / std::sqrt(static_cast<double>(p_.dim));
      for (double& a : proj_) a = rng_.normal() * scale;  // E||Az||^2 = ||z||^2
    } else if (p_.type != "uniform") {
      throw hnsw::InvalidArgument("--type must be uniform or clustered");
    }
  }

  /// Fills `count` rows using a separate stream so that base and query sets are independent
  /// but share the same mixture (centres and projection).
  hnsw::Matrix sample(std::size_t count, std::uint64_t stream_seed) {
    bench::Rng rng(stream_seed);
    hnsw::Matrix m;
    m.rows = count;
    m.cols = p_.dim;
    m.data.resize(count * p_.dim);
    std::vector<double> z(r_);
    for (std::size_t i = 0; i < count; ++i) {
      float* x = m.row(i);
      if (p_.type == "uniform") {
        for (std::size_t j = 0; j < p_.dim; ++j) x[j] = static_cast<float>(rng.uniform());
        continue;
      }
      const std::size_t c = static_cast<std::size_t>(rng.below(p_.clusters));
      for (std::size_t t = 0; t < r_; ++t) z[t] = centers_[c * r_ + t] + p_.spread * rng.normal();
      for (std::size_t j = 0; j < p_.dim; ++j) {
        double acc = 0.0;
        for (std::size_t t = 0; t < r_; ++t) acc += proj_[j * r_ + t] * z[t];
        x[j] = static_cast<float>(acc + p_.noise * rng.normal());
      }
    }
    return m;
  }

 private:
  GenParams p_;
  bench::Rng rng_;
  std::size_t r_ = 0;
  std::vector<double> centers_;
  std::vector<double> proj_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const bench::ArgMap args(argc, argv, 1);
    args.allow({"out", "type", "n", "nq", "dim", "clusters", "latent", "spread", "noise", "seed", "metric", "gt-k",
                "threads", "csv"});
    const std::string out = args.required("out");
    GenParams p;
    p.type = args.str("type", "clustered");
    p.dim = args.size("dim", 128);
    p.clusters = std::max<std::size_t>(1, args.size("clusters", 64));
    p.latent = std::max<std::size_t>(1, args.size("latent", 16));
    p.spread = args.real("spread", 0.1);
    p.noise = args.real("noise", 0.01);
    const std::size_t n = args.size("n", 10000);
    const std::size_t nq = args.size("nq", 1000);
    const std::uint64_t seed = args.u64("seed", 1);
    const hnsw::Metric metric = hnsw::parse_metric(args.str("metric", "l2"));
    const std::size_t gt_k = args.size("gt-k", 100);
    const unsigned threads = static_cast<unsigned>(args.size("threads", bench::default_threads()));
    if (p.dim == 0 || p.dim > hnsw::kMaxDim || n == 0) throw hnsw::InvalidArgument("bad --dim or --n");

    hnsw::Timer t;
    Generator gen(p, seed);
    const hnsw::Matrix base = gen.sample(n, seed * 1000003ULL + 1);
    const hnsw::Matrix queries = gen.sample(nq, seed * 1000003ULL + 2);
    hnsw::write_fvecs(out + "_base.fvecs", base);
    hnsw::write_fvecs(out + "_query.fvecs", queries);
    const double gen_s = t.seconds();

    t.restart();
    const auto gt = bench::exact_neighbors(metric, base, {}, queries, std::min(gt_k, n), threads);
    std::vector<std::vector<std::int32_t>> rows;
    for (const auto& r : gt) rows.emplace_back(r.begin(), r.end());
    hnsw::write_ivecs(out + "_gt.ivecs", rows);
    const double gt_s = t.seconds();

    if (args.has("csv")) {
      hnsw::LabeledVectors lv;
      lv.vectors = base;
      for (std::size_t i = 0; i < n; ++i) lv.ids.push_back(i);
      hnsw::write_csv_vectors(out + "_base.csv", lv);
      lv.vectors = queries;
      lv.ids.clear();
      for (std::size_t i = 0; i < nq; ++i) lv.ids.push_back(i);
      hnsw::write_csv_vectors(out + "_query.csv", lv);
    }

    std::ofstream meta(out + "_meta.txt");
    meta << "type=" << p.type << "\nn=" << n << "\nnq=" << nq << "\ndim=" << p.dim << "\nclusters=" << p.clusters
         << "\nlatent=" << p.latent << "\nspread=" << p.spread << "\nnoise=" << p.noise << "\nseed=" << seed
         << "\nmetric=" << hnsw::metric_name(metric) << "\ngt_k=" << gt_k << "\n";
    std::cout << "wrote " << out << "_{base,query}.fvecs and _gt.ivecs: n=" << n << " nq=" << nq << " d=" << p.dim
              << " type=" << p.type << " (generate " << bench::fmt(gen_s, 3) << " s, ground truth "
              << bench::fmt(gt_s, 3) << " s on " << threads << " threads)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
