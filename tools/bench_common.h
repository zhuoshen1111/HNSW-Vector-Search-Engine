// Helpers shared by the data generator and the benchmark driver (not part of the library).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "hnsw/common.h"
#include "hnsw/index.h"
#include "hnsw/io.h"

namespace bench {

/// "--key value" / "--flag" command-line parser with typed getters and unknown-option checks.
class ArgMap {
 public:
  ArgMap(int argc, char** argv, int first);
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def) const;
  std::string required(const std::string& k) const;
  std::size_t size(const std::string& k, std::size_t def) const;
  std::uint64_t u64(const std::string& k, std::uint64_t def) const;
  double real(const std::string& k, double def) const;
  std::vector<std::size_t> sizes(const std::string& k, const std::vector<std::size_t>& def) const;
  std::vector<double> reals(const std::string& k, const std::vector<double>& def) const;
  std::vector<std::string> strs(const std::string& k, const std::vector<std::string>& def) const;
  /// Throws if an option outside `allowed` was given.
  void allow(const std::vector<std::string>& allowed) const;

 private:
  std::map<std::string, std::string> kv_;
};

/// Platform-independent RNG helpers on top of std::mt19937_64 (std distributions are not
/// bit-identical across standard libraries, so generated datasets would differ).
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : eng_(seed) {}
  std::uint64_t next() { return eng_(); }
  double uniform() { return static_cast<double>(eng_() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
  std::uint64_t below(std::uint64_t n) { return n == 0 ? 0 : next() % n; }
  double normal();  // Box-Muller
 private:
  std::mt19937_64 eng_;
  bool have_spare_ = false;
  double spare_ = 0.0;
};

/// Exact top-k ids of each query under the index metric, computed with `threads` threads by a
/// BruteForceIndex holding `base` (row i has id ids[i], or i when ids is empty).
std::vector<std::vector<hnsw::label_t>> exact_neighbors(hnsw::Metric metric, const hnsw::Matrix& base,
                                                        const std::vector<hnsw::label_t>& ids,
                                                        const hnsw::Matrix& queries, std::size_t k,
                                                        unsigned threads);

/// Mean over queries of |result ∩ truth[:k]| / min(k, |truth|).
double recall_at_k(const std::vector<std::vector<hnsw::SearchResult>>& results,
                   const std::vector<std::vector<hnsw::label_t>>& truth, std::size_t k);

/// Peak resident set size of this process in bytes (getrusage; 0 if unavailable).
std::size_t peak_rss_bytes();
/// Current resident set size in bytes where the OS reports it, else 0.
std::size_t current_rss_bytes();

std::string machine_description();
unsigned default_threads();

/// Appends rows to a CSV file, writing the header when the file is new or empty.
class CsvWriter {
 public:
  CsvWriter(const std::string& path, const std::vector<std::string>& header);
  void row(const std::vector<std::string>& cells);

 private:
  std::string path_;
  std::size_t columns_;
};

std::string fmt(double v, int digits = 6);

}  // namespace bench
