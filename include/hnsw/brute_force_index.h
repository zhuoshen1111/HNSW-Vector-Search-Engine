// BruteForceIndex<Dist>: exact k-NN by exhaustive scan. It is the correctness reference for the
// graph index and the "no index" baseline of the experiments.
//
// Storage: contiguous VectorStore + parallel label array + OpenHashMap(id -> slot).
// remove() is swap-with-last, so the store stays dense (no tombstones).
//
// Complexity (n live vectors, d dimensions):
//   add            O(d) amortised          remove        O(d) expected
//   search         O(n*d + n log k) worst case with Heap (the default; most candidates are
//                  rejected by one comparison with the heap top), O(n*d + n + k log k) expected
//                  with QuickSelect, O(n*d + n log n) with FullSort.      memory O(n*d)
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/common.h"
#include "hnsw/distance.h"
#include "hnsw/hash_table.h"
#include "hnsw/heap.h"
#include "hnsw/index.h"
#include "hnsw/serializer.h"
#include "hnsw/top_k.h"
#include "hnsw/vector_store.h"

namespace hnsw {

/// Strategy for picking the k best of n exact distances (compared in the top-k microbenchmark).
enum class TopKMethod : std::uint8_t { QuickSelect = 0, Heap = 1, FullSort = 2 };
const char* topk_method_name(TopKMethod m) noexcept;
TopKMethod parse_topk_method(const std::string& s);

template <typename Dist>
class BruteForceIndex final : public Index {
 public:
  explicit BruteForceIndex(std::size_t dim) : store_(dim) {}

  IndexKind kind() const noexcept override { return IndexKind::BruteForce; }
  Metric metric() const noexcept override { return Dist::kMetric; }
  std::size_t dim() const noexcept override { return store_.dim(); }
  std::size_t size() const noexcept override { return labels_.size(); }
  bool contains(label_t id) const override { return id_map_.contains(id); }

  void validate_vector(const float* v, std::size_t d) const override {
    check_vector_common(v, d, dim());
    Dist::check(v, d);
  }

  void add(label_t id, const float* v, std::size_t d) override {
    validate_vector(v, d);
    if (id_map_.contains(id)) throw InvalidArgument("id " + std::to_string(id) + " already exists");
    if (labels_.size() >= static_cast<std::size_t>(kInvalidSlot) - 1) throw InvalidArgument("index is full");
    std::vector<float> buf(v, v + d);
    Dist::prepare(buf.data(), d);
    add_prepared(id, buf.data());
  }

  void remove(label_t id) override {
    const slot_t* found = id_map_.find(id);
    if (found == nullptr) throw NotFound("id " + std::to_string(id) + " is not in the index");
    const slot_t s = *found;
    const slot_t last = static_cast<slot_t>(labels_.size() - 1);
    id_map_.erase(id);
    if (s != last) {
      store_.set(s, store_.get(last));
      labels_[s] = labels_[last];
      *id_map_.find(labels_[s]) = s;
    }
    store_.pop_back();
    labels_.pop_back();
  }

  std::vector<SearchResult> search(const float* q, std::size_t d, std::size_t k, const SearchParams& /*params*/,
                                   SearchStats* stats) const override {
    validate_vector(q, d);
    const std::size_t n = labels_.size();
    if (k == 0 || n == 0) return {};
    std::vector<float> qbuf(q, q + d);
    Dist::prepare(qbuf.data(), d);
    const float* qp = qbuf.data();
    if (stats) {
      stats->distance_computations += n;
      stats->visited += n;
    }
    k = std::min(k, n);
    if (method_ == TopKMethod::Heap) {
      BoundedMaxHeap<SearchResult, ResultOrder> best(k);
      for (std::size_t i = 0; i < n; ++i) {
        best.push(SearchResult{labels_[i], Dist::distance(qp, store_.get(static_cast<slot_t>(i)), d)});
      }
      return best.take_sorted();
    }
    std::vector<SearchResult> all(n);
    for (std::size_t i = 0; i < n; ++i) {
      all[i] = SearchResult{labels_[i], Dist::distance(qp, store_.get(static_cast<slot_t>(i)), d)};
    }
    if (method_ == TopKMethod::QuickSelect) {
      select_smallest(all.begin(), n, k, ResultOrder{});
      all.resize(k);
      std::sort(all.begin(), all.end(), ResultOrder{});
    } else {
      std::sort(all.begin(), all.end(), ResultOrder{});
      all.resize(k);
    }
    return all;
  }

  void reserve(std::size_t n) override {
    store_.reserve(n);
    labels_.reserve(n);
    id_map_.reserve(n);
  }

  void compact() override {}  // already dense

  std::vector<label_t> labels() const override { return labels_; }

  bool get_vector(label_t id, std::vector<float>& out) const override {
    const slot_t* s = id_map_.find(id);
    if (s == nullptr) return false;
    const float* p = store_.get(*s);
    out.assign(p, p + dim());
    return true;
  }

  IndexStats stats() const override {
    IndexStats st;
    st.kind = kind_name(kind());
    st.metric = metric_name(metric());
    st.dim = dim();
    st.live = labels_.size();
    st.vector_bytes = store_.memory_bytes();
    st.id_map_bytes = id_map_.memory_bytes();
    st.metadata_bytes = labels_.capacity() * sizeof(label_t);
    st.params.emplace_back("topk", topk_method_name(method_));
    return st;
  }

  std::vector<std::string> check_invariants() const override {
    std::vector<std::string> errs;
    if (store_.size() != labels_.size()) errs.push_back("vector count differs from label count");
    if (id_map_.size() != labels_.size()) errs.push_back("id map size differs from label count");
    std::string why;
    if (!id_map_.check_invariants(&why)) errs.push_back("id map: " + why);
    for (std::size_t i = 0; i < labels_.size() && errs.size() < 20; ++i) {
      const slot_t* s = id_map_.find(labels_[i]);
      if (s == nullptr || *s != i) errs.push_back("id map does not point back to slot " + std::to_string(i));
    }
    return errs;
  }

  std::string describe() const override {
    return std::string("bruteforce(") + metric_name(metric()) + ", d=" + std::to_string(dim()) + ")";
  }

  void save_body(BinaryWriter& w) const override {
    w.write_pod<std::uint64_t>(labels_.size());
    w.write_array(labels_);
    w.write_array(store_.raw());
  }

  static std::unique_ptr<BruteForceIndex> load_body(BinaryReader& r, std::size_t dim) {
    auto idx = std::make_unique<BruteForceIndex>(dim);
    const std::uint64_t n = r.read_u64_in_range("vector count", 0, kInvalidSlot - 1);
    idx->labels_ = r.read_array<label_t>(n);
    std::vector<float> raw = r.read_array<float>(n * dim);
    for (float x : raw) {
      if (!std::isfinite(x)) throw FormatError("index file contains a non-finite vector component");
    }
    idx->store_ = VectorStore::from_raw(dim, std::move(raw));
    idx->id_map_.reserve(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < idx->labels_.size(); ++i) {
      if (!idx->id_map_.insert(idx->labels_[i], static_cast<slot_t>(i))) {
        throw FormatError("index file contains duplicate id " + std::to_string(idx->labels_[i]));
      }
    }
    return idx;
  }

  void set_topk_method(TopKMethod m) noexcept { method_ = m; }
  TopKMethod topk_method() const noexcept { return method_; }

  /// Adds a vector that has already been validated and prepared.
  void add_prepared(label_t id, const float* prepared) {
    const slot_t s = store_.add(prepared);
    labels_.push_back(id);
    id_map_.insert(id, s);
  }

 private:
  VectorStore store_;
  std::vector<label_t> labels_;
  OpenHashMap<label_t, slot_t> id_map_;
  TopKMethod method_ = TopKMethod::Heap;  // fastest in the e9 microbenchmark for k <= 100
};

extern template class BruteForceIndex<L2>;
extern template class BruteForceIndex<InnerProduct>;
extern template class BruteForceIndex<Cosine>;

}  // namespace hnsw
