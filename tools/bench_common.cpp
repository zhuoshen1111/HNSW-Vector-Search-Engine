#include "bench_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include <sys/resource.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include "hnsw/brute_force_index.h"
#include "hnsw/index_factory.h"

namespace bench {

ArgMap::ArgMap(int argc, char** argv, int first) {
  for (int i = first; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() < 3 || a.compare(0, 2, "--") != 0) throw hnsw::InvalidArgument("unexpected argument '" + a + "'");
    a = a.substr(2);
    std::string v = "1";
    if (i + 1 < argc && std::string(argv[i + 1]).compare(0, 2, "--") != 0) v = argv[++i];
    kv_[a] = v;
  }
}

std::string ArgMap::str(const std::string& k, const std::string& def) const {
  const auto it = kv_.find(k);
  return it == kv_.end() ? def : it->second;
}

std::string ArgMap::required(const std::string& k) const {
  const auto it = kv_.find(k);
  if (it == kv_.end()) throw hnsw::InvalidArgument("missing required option --" + k);
  return it->second;
}

std::uint64_t ArgMap::u64(const std::string& k, std::uint64_t def) const {
  const auto it = kv_.find(k);
  if (it == kv_.end()) return def;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(it->second.c_str(), &end, 10);
  if (it->second.empty() || *end != '\0') throw hnsw::InvalidArgument("--" + k + " expects an integer");
  return v;
}

std::size_t ArgMap::size(const std::string& k, std::size_t def) const { return static_cast<std::size_t>(u64(k, def)); }

double ArgMap::real(const std::string& k, double def) const {
  const auto it = kv_.find(k);
  if (it == kv_.end()) return def;
  char* end = nullptr;
  const double v = std::strtod(it->second.c_str(), &end);
  if (it->second.empty() || *end != '\0') throw hnsw::InvalidArgument("--" + k + " expects a number");
  return v;
}

std::vector<std::string> ArgMap::strs(const std::string& k, const std::vector<std::string>& def) const {
  const auto it = kv_.find(k);
  if (it == kv_.end()) return def;
  std::vector<std::string> out;
  std::stringstream ss(it->second);
  std::string t;
  while (std::getline(ss, t, ',')) {
    if (!t.empty()) out.push_back(t);
  }
  return out;
}

std::vector<std::size_t> ArgMap::sizes(const std::string& k, const std::vector<std::size_t>& def) const {
  if (!has(k)) return def;
  std::vector<std::size_t> out;
  for (const std::string& s : strs(k, {})) out.push_back(static_cast<std::size_t>(std::stoull(s)));
  return out;
}

std::vector<double> ArgMap::reals(const std::string& k, const std::vector<double>& def) const {
  if (!has(k)) return def;
  std::vector<double> out;
  for (const std::string& s : strs(k, {})) out.push_back(std::stod(s));
  return out;
}

void ArgMap::allow(const std::vector<std::string>& allowed) const {
  for (const auto& kv : kv_) {
    if (std::find(allowed.begin(), allowed.end(), kv.first) == allowed.end()) {
      throw hnsw::InvalidArgument("unknown option --" + kv.first);
    }
  }
}

double Rng::normal() {
  if (have_spare_) {
    have_spare_ = false;
    return spare_;
  }
  double u1 = uniform();
  while (u1 <= 1e-300) u1 = uniform();
  const double u2 = uniform();
  const double r = std::sqrt(-2.0 * std::log(u1));
  const double theta = 2.0 * 3.14159265358979323846 * u2;
  spare_ = r * std::sin(theta);
  have_spare_ = true;
  return r * std::cos(theta);
}

std::vector<std::vector<hnsw::label_t>> exact_neighbors(hnsw::Metric metric, const hnsw::Matrix& base,
                                                        const std::vector<hnsw::label_t>& ids,
                                                        const hnsw::Matrix& queries, std::size_t k,
                                                        unsigned threads) {
  hnsw::IndexConfig cfg;
  cfg.kind = hnsw::IndexKind::BruteForce;
  cfg.metric = metric;
  cfg.dim = base.cols;
  std::unique_ptr<hnsw::Index> bf = hnsw::make_index(cfg);
  bf->reserve(base.rows);
  for (std::size_t i = 0; i < base.rows; ++i) bf->add(ids.empty() ? i : ids[i], base.row(i), base.cols);
  std::vector<std::vector<hnsw::label_t>> out(queries.rows);
  threads = std::max(1u, threads);
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t]() {
      hnsw::SearchParams p;
      for (std::size_t q = t; q < queries.rows; q += threads) {
        // BruteForceIndex::search is const and has no mutable state, so concurrent calls are safe.
        for (const hnsw::SearchResult& r : bf->search(queries.row(q), queries.cols, k, p, nullptr)) out[q].push_back(r.id);
      }
    });
  }
  for (std::thread& th : pool) th.join();
  return out;
}

double recall_at_k(const std::vector<std::vector<hnsw::SearchResult>>& results,
                   const std::vector<std::vector<hnsw::label_t>>& truth, std::size_t k) {
  if (results.empty()) return 0.0;
  double total = 0.0;
  for (std::size_t q = 0; q < results.size(); ++q) {
    const std::vector<hnsw::label_t>& t = truth[q];
    const std::size_t denom = std::min(k, t.size());
    if (denom == 0) {
      total += 1.0;
      continue;
    }
    std::size_t hit = 0;
    for (std::size_t i = 0; i < results[q].size() && i < k; ++i) {
      if (std::find(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(denom), results[q][i].id) !=
          t.begin() + static_cast<std::ptrdiff_t>(denom)) {
        ++hit;
      }
    }
    total += static_cast<double>(hit) / static_cast<double>(denom);
  }
  return total / static_cast<double>(results.size());
}

std::size_t peak_rss_bytes() {
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::size_t>(ru.ru_maxrss);  // bytes on macOS
#else
  return static_cast<std::size_t>(ru.ru_maxrss) * 1024;  // kilobytes on Linux
#endif
}

std::size_t current_rss_bytes() {
#if defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<std::size_t>(info.resident_size);
#else
  std::ifstream f("/proc/self/statm");
  std::size_t pages_total = 0, pages_rss = 0;
  if (!(f >> pages_total >> pages_rss)) return 0;
  return pages_rss * 4096;
#endif
}

std::string machine_description() {
  std::string cpu = "unknown-cpu";
#if defined(__APPLE__)
  char buf[256];
  std::size_t len = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) cpu = std::string(buf);
#else
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 10, "model name") == 0) {
      cpu = line.substr(line.find(':') + 2);
      break;
    }
  }
#endif
  return cpu + ", " + std::to_string(std::thread::hardware_concurrency()) + " hw threads";
}

unsigned default_threads() { return std::max(1u, std::thread::hardware_concurrency()); }

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header)
    : path_(path), columns_(header.size()) {
  bool fresh = true;
  {
    std::ifstream in(path);
    if (in && in.peek() != std::ifstream::traits_type::eof()) fresh = false;
  }
  if (fresh) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw hnsw::IoError("cannot write '" + path + "'");
    for (std::size_t i = 0; i < header.size(); ++i) out << (i ? "," : "") << header[i];
    out << "\n";
  }
}

void CsvWriter::row(const std::vector<std::string>& cells) {
  if (cells.size() != columns_) throw hnsw::InvalidArgument("CSV row has the wrong number of columns");
  std::ofstream out(path_, std::ios::app);
  if (!out) throw hnsw::IoError("cannot append to '" + path_ + "'");
  for (std::size_t i = 0; i < cells.size(); ++i) out << (i ? "," : "") << cells[i];
  out << "\n";
}

std::string fmt(double v, int digits) {
  std::ostringstream os;
  os << std::setprecision(digits) << v;
  return os.str();
}

}  // namespace bench
