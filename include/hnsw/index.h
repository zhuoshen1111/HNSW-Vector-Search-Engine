// The polymorphic Index interface shared by the brute-force reference and the HNSW graph.
//
// Contract (both implementations):
//   * add(id, v): v must have dim() finite floats (and be non-zero under cosine); id must not be
//     live. On any violation an exception is thrown and the index is unchanged.
//   * remove(id): id must be live, otherwise NotFound and the index is unchanged. A removed id is
//     never returned again and may later be re-added.
//   * search(q, k): returns at most k live results ordered by (distance, id). k == 0 or an empty
//     index gives an empty answer; k > size() returns every reachable live vector.
//     BruteForceIndex is exact. HNSWIndex is approximate; ef controls its accuracy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/common.h"
#include "hnsw/distance.h"

namespace hnsw {

class BinaryWriter;

enum class IndexKind : std::uint8_t { BruteForce = 0, HNSW = 1 };
const char* kind_name(IndexKind k) noexcept;

struct SearchParams {
  /// HNSW search breadth (size of the dynamic result list). The effective value is max(ef, k).
  /// Ignored by the brute-force index.
  std::size_t ef = 50;
};

/// Work counters accumulated by search() when a non-null pointer is passed.
struct SearchStats {
  std::uint64_t distance_computations = 0;
  std::uint64_t visited = 0;   ///< nodes inserted into the visited set
  std::uint64_t expanded = 0;  ///< candidates whose adjacency list was scanned
  SearchStats& operator+=(const SearchStats& o) noexcept {
    distance_computations += o.distance_computations;
    visited += o.visited;
    expanded += o.expanded;
    return *this;
  }
};

/// Structural and memory statistics. Byte counts are explicit estimates of the data structures'
/// reserved storage (vector capacities), not process RSS.
struct IndexStats {
  std::string kind;
  std::string metric;
  std::size_t dim = 0;
  std::size_t live = 0;
  std::size_t tombstones = 0;  ///< deleted slots still stored (HNSW only)
  int max_level = -1;
  bool has_entry = false;
  label_t entry_id = 0;
  bool entry_deleted = false;
  std::vector<std::size_t> nodes_per_level;  ///< nodes (live + tombstone) present at each level
  std::vector<double> avg_out_degree;        ///< mean out-degree of live nodes per level
  std::size_t edges = 0;                     ///< out-edges of live nodes, all levels
  std::size_t edges_to_tombstones = 0;       ///< of those, edges pointing at deleted nodes
  std::size_t unreachable_live = 0;          ///< live nodes not reachable on layer 0 from the entry
  std::size_t vector_bytes = 0;
  std::size_t graph_bytes = 0;
  std::size_t id_map_bytes = 0;
  std::size_t metadata_bytes = 0;  ///< labels, levels, flags, visited table
  std::vector<std::pair<std::string, std::string>> params;

  std::size_t total_bytes() const noexcept { return vector_bytes + graph_bytes + id_map_bytes + metadata_bytes; }
  double bytes_per_live_vector() const noexcept {
    return live == 0 ? 0.0 : static_cast<double>(total_bytes()) / static_cast<double>(live);
  }
};

/// Human-readable multi-line rendering used by the CLI `stats` command.
std::string format_stats(const IndexStats& s);

/// Throws InvalidArgument if v is null, has the wrong dimension, or contains NaN/inf.
void check_vector_common(const float* v, std::size_t d, std::size_t expected_dim);

class Index {
 public:
  virtual ~Index() = default;
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  Index(Index&&) = delete;
  Index& operator=(Index&&) = delete;

  virtual IndexKind kind() const noexcept = 0;
  virtual Metric metric() const noexcept = 0;
  virtual std::size_t dim() const noexcept = 0;
  /// Number of live (searchable) vectors.
  virtual std::size_t size() const noexcept = 0;
  virtual bool contains(label_t id) const = 0;

  /// Throws exactly when add() would reject v for vector-content reasons (used to pre-validate
  /// batches so that a failing batch leaves the index unchanged).
  virtual void validate_vector(const float* v, std::size_t d) const = 0;
  virtual void add(label_t id, const float* v, std::size_t d) = 0;
  virtual void remove(label_t id) = 0;
  virtual std::vector<SearchResult> search(const float* q, std::size_t d, std::size_t k,
                                           const SearchParams& params, SearchStats* stats) const = 0;

  virtual void reserve(std::size_t n) = 0;
  /// Physically drops deleted vectors (HNSW: rebuilds the graph from the live set).
  virtual void compact() = 0;
  /// Live ids in internal slot order.
  virtual std::vector<label_t> labels() const = 0;
  /// Copies the stored (metric-prepared) vector of a live id. Returns false if id is not live.
  virtual bool get_vector(label_t id, std::vector<float>& out) const = 0;
  virtual IndexStats stats() const = 0;
  /// Checks every structural invariant; returns human-readable violations (empty = healthy).
  virtual std::vector<std::string> check_invariants() const = 0;
  /// One-line description, e.g. "hnsw(l2, d=128, M=16, efc=200, heuristic, repair)".
  virtual std::string describe() const = 0;
  /// Writes the kind-specific body of the index file (the header is written by save_index).
  virtual void save_body(BinaryWriter& w) const = 0;

  bool empty() const noexcept { return size() == 0; }
  /// Convenience wrapper around search().
  std::vector<SearchResult> knn(const std::vector<float>& q, std::size_t k, std::size_t ef = 50) const {
    SearchParams p;
    p.ef = ef;
    return search(q.data(), q.size(), k, p, nullptr);
  }

 protected:
  Index() = default;
};

}  // namespace hnsw
