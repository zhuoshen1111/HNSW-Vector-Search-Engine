#include "hnsw/index.h"

#include <cmath>
#include <iomanip>
#include <sstream>

#include "hnsw/brute_force_index.h"

namespace hnsw {

const char* kind_name(IndexKind k) noexcept {
  switch (k) {
    case IndexKind::BruteForce:
      return "bruteforce";
    case IndexKind::HNSW:
      return "hnsw";
  }
  return "unknown";
}

void check_vector_common(const float* v, std::size_t d, std::size_t expected_dim) {
  if (v == nullptr) throw InvalidArgument("null vector");
  if (d != expected_dim) {
    throw InvalidArgument("dimension mismatch: vector has " + std::to_string(d) + " components, index expects " +
                          std::to_string(expected_dim));
  }
  for (std::size_t i = 0; i < d; ++i) {
    if (!std::isfinite(v[i])) {
      throw InvalidArgument("vector component " + std::to_string(i) + " is NaN or infinite");
    }
  }
}

const char* topk_method_name(TopKMethod m) noexcept {
  switch (m) {
    case TopKMethod::QuickSelect:
      return "quickselect";
    case TopKMethod::Heap:
      return "heap";
    case TopKMethod::FullSort:
      return "fullsort";
  }
  return "unknown";
}

TopKMethod parse_topk_method(const std::string& s) {
  if (s == "quickselect" || s == "select") return TopKMethod::QuickSelect;
  if (s == "heap") return TopKMethod::Heap;
  if (s == "fullsort" || s == "sort") return TopKMethod::FullSort;
  throw InvalidArgument("unknown top-k method '" + s + "' (expected quickselect, heap or fullsort)");
}

namespace {
std::string human_bytes(std::size_t b) {
  std::ostringstream os;
  const double x = static_cast<double>(b);
  if (x >= 1024.0 * 1024.0 * 1024.0) {
    os << std::fixed << std::setprecision(2) << x / (1024.0 * 1024.0 * 1024.0) << " GiB";
  } else if (x >= 1024.0 * 1024.0) {
    os << std::fixed << std::setprecision(2) << x / (1024.0 * 1024.0) << " MiB";
  } else if (x >= 1024.0) {
    os << std::fixed << std::setprecision(2) << x / 1024.0 << " KiB";
  } else {
    os << b << " B";
  }
  return os.str();
}
}  // namespace

std::string format_stats(const IndexStats& s) {
  std::ostringstream os;
  os << "kind:            " << s.kind << "\n";
  os << "metric:          " << s.metric << "\n";
  os << "dimension:       " << s.dim << "\n";
  os << "live vectors:    " << s.live << "\n";
  if (s.kind != "bruteforce") {
    os << "tombstones:      " << s.tombstones << "\n";
    os << "max level:       " << s.max_level << "\n";
    if (s.has_entry) {
      os << "entry point:     id " << s.entry_id << (s.entry_deleted ? " (deleted)" : "") << "\n";
    }
    os << "nodes per level: ";
    for (std::size_t l = 0; l < s.nodes_per_level.size(); ++l) os << (l ? " " : "") << s.nodes_per_level[l];
    os << "\n";
    os << "avg out-degree:  ";
    for (std::size_t l = 0; l < s.avg_out_degree.size(); ++l) {
      os << (l ? " " : "") << std::fixed << std::setprecision(2) << s.avg_out_degree[l];
    }
    os << "\n";
    os << "edges:           " << s.edges << " (" << s.edges_to_tombstones << " to tombstones)\n";
    os << "unreachable:     " << s.unreachable_live << " live vectors not reachable on layer 0\n";
  }
  for (const auto& kv : s.params) os << std::left << std::setw(17) << (kv.first + ":") << kv.second << "\n";
  os << "memory:          " << human_bytes(s.total_bytes()) << " (vectors " << human_bytes(s.vector_bytes)
     << ", graph " << human_bytes(s.graph_bytes) << ", id map " << human_bytes(s.id_map_bytes) << ", other "
     << human_bytes(s.metadata_bytes) << ")\n";
  os << "bytes/vector:    " << std::fixed << std::setprecision(1) << s.bytes_per_live_vector() << "\n";
  return os.str();
}

}  // namespace hnsw
