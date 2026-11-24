// HNSWIndex<Dist>: Hierarchical Navigable Small World graph (Malkov & Yashunin, TPAMI 2018).
//
// Algorithms implemented here (paper numbering):
//   Alg. 1 INSERT            random level, greedy descent, per-layer beam search, linking
//   Alg. 2 SEARCH-LAYER      best-first beam search with a candidate min-heap and a bounded
//                            result max-heap of size ef
//   Alg. 3 SELECT-SIMPLE     keep the M closest candidates
//   Alg. 4 SELECT-HEURISTIC  keep a candidate only if it is closer to the base point than to
//                            every neighbour already kept (diversifies directions)
//   Alg. 5 K-NN-SEARCH       greedy descent through upper layers, beam search on layer 0
// plus our deletion layer: tombstones (deleted nodes stay routable but are never returned) and
// optional local neighbour repair, entry-point re-election and full compaction.
//
// Data layout (n = slots including tombstones, M0 = layer-0 degree bound, M = upper bound):
//   store_    n*d floats, contiguous           labels_/levels_/deleted_  one entry per slot
//   links0_   n*(M0+1) u32: [count, nbr_1..nbr_M0] per slot, fixed stride
//   upper_    per slot, level*(M+1) u32 for layers 1..level (empty for most nodes)
//   id_map_   OpenHashMap external id -> slot (live ids only)
// Edges are 32-bit slot numbers, never pointers, so growth/reallocation and save/load are safe.
//
// Thread safety: search() is logically const but uses mutable scratch (visited table, heaps);
// concurrent calls on one index are not supported.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/common.h"
#include "hnsw/distance.h"
#include "hnsw/hash_table.h"
#include "hnsw/heap.h"
#include "hnsw/index.h"
#include "hnsw/serializer.h"
#include "hnsw/vector_store.h"
#include "hnsw/visited_table.h"

namespace hnsw {

enum class NeighborSelection : std::uint8_t { Simple = 0, Heuristic = 1 };
/// Tombstone: mark only. The repair variants re-select the lists of the deleted node's neighbours
/// from N(u) ∪ N(x) with the heuristic and differ only in the resulting list length:
///   RepairPrune  keep what the heuristic keeps (lists may shrink);
///   Repair       never shrink: refill to the previous length with the closest pruned candidates
///                (the heuristic may still keep more) -- the default;
///   RepairFixed  exactly the previous length (refill, then truncate) -- an ablation showing that
///                truncation cuts long-range links and strands nodes (report E6).
enum class DeletePolicy : std::uint8_t { Tombstone = 0, RepairPrune = 1, Repair = 2, RepairFixed = 3 };

const char* selection_name(NeighborSelection s) noexcept;
NeighborSelection parse_selection(const std::string& s);
const char* delete_policy_name(DeletePolicy p) noexcept;
DeletePolicy parse_delete_policy(const std::string& s);

struct HnswParams {
  std::size_t M = 16;                ///< out-degree bound on layers >= 1 and links per new node
  std::size_t M0 = 0;                ///< layer-0 degree bound; 0 means 2*M (paper default)
  std::size_t ef_construction = 200; ///< beam width while inserting
  std::uint64_t seed = 42;           ///< level-assignment RNG seed (builds are deterministic)
  NeighborSelection selection = NeighborSelection::Heuristic;
  DeletePolicy delete_policy = DeletePolicy::Repair;
  bool single_layer = false;         ///< NSW baseline: every node on layer 0 only
  /// Paper Alg. 4 "keepPrunedConnections": after the heuristic, refill the list up to M with the
  /// closest pruned candidates. Always enabled for non-metric distances (inner product), where
  /// the heuristic otherwise collapses the out-degree to ~2 and disconnects the graph.
  bool keep_pruned = false;

  std::size_t effective_M0() const noexcept { return M0 == 0 ? 2 * M : M0; }
  /// Throws InvalidArgument for out-of-range values.
  void validate() const;
};

/// Skip-list style level sampler: level = floor(-ln(U) * mult), U ~ Uniform(0, 1], mult = 1/ln(M).
/// P(level >= l) = M^-l, so the expected number of nodes on layer l shrinks geometrically.
/// Uses std::mt19937_64 (bit-exact across platforms) and its own uniform conversion.
class LevelGenerator {
 public:
  static constexpr int kMaxLevel = 30;
  LevelGenerator(std::uint64_t seed, double mult) : rng_(seed), mult_(mult) {}
  int next() noexcept;
  std::string state() const;
  /// Throws FormatError if s is not a valid engine state.
  void restore(const std::string& s);

 private:
  std::mt19937_64 rng_;
  double mult_;
};

template <typename Dist>
class HNSWIndex final : public Index {
 public:
  HNSWIndex(std::size_t dim, const HnswParams& params);

  // ---- Index interface ----
  IndexKind kind() const noexcept override { return IndexKind::HNSW; }
  Metric metric() const noexcept override { return Dist::kMetric; }
  std::size_t dim() const noexcept override { return dim_; }
  std::size_t size() const noexcept override { return live_; }
  bool contains(label_t id) const override { return id_map_.contains(id); }
  void validate_vector(const float* v, std::size_t d) const override {
    check_vector_common(v, d, dim_);
    Dist::check(v, d);
  }
  void add(label_t id, const float* v, std::size_t d) override;
  void remove(label_t id) override;
  std::vector<SearchResult> search(const float* q, std::size_t d, std::size_t k, const SearchParams& params,
                                   SearchStats* stats) const override;
  void reserve(std::size_t n) override;
  void compact() override;
  std::vector<label_t> labels() const override;
  bool get_vector(label_t id, std::vector<float>& out) const override;
  IndexStats stats() const override;
  std::vector<std::string> check_invariants() const override;
  std::string describe() const override;
  void save_body(BinaryWriter& w) const override;

  static std::unique_ptr<HNSWIndex> load_body(BinaryReader& r, std::size_t dim);

  // ---- Inspection helpers (tests, CLI) ----
  const HnswParams& params() const noexcept { return params_; }
  std::size_t slot_count() const noexcept { return labels_.size(); }
  std::size_t tombstone_count() const noexcept { return labels_.size() - live_; }
  int max_level() const noexcept { return max_level_; }
  /// Top level of a live id, or -1 if the id is not live.
  int level_of(label_t id) const;
  /// External ids of the out-neighbours of a live id on a layer (deleted neighbours included).
  std::vector<label_t> neighbors_of(label_t id, int level) const;
  /// External id of the entry point. Precondition: slot_count() > 0.
  label_t entry_id() const noexcept { return labels_[entry_]; }
  /// In-degree on layer 0 of every live id (for hub experiments).
  std::vector<std::pair<label_t, std::size_t>> layer0_in_degrees() const;
  /// Adds a vector that has already passed validate_vector() and Dist::prepare().
  void add_prepared(label_t id, const float* prepared);

 private:
  struct Candidate {
    float dist;
    slot_t slot;
    label_t label;
  };
  /// (distance, external id, slot): the same total order the brute-force reference uses.
  struct CandidateLess {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept {
      if (a.dist < b.dist) return true;
      if (b.dist < a.dist) return false;
      if (a.label != b.label) return a.label < b.label;
      return a.slot < b.slot;
    }
  };

  std::size_t max_degree(int level) const noexcept { return level == 0 ? maxM0_ : maxM_; }
  const slot_t* list(slot_t s, int level) const noexcept {
    return level == 0 ? links0_.data() + static_cast<std::size_t>(s) * (maxM0_ + 1)
                      : upper_[s].data() + static_cast<std::size_t>(level - 1) * (maxM_ + 1);
  }
  slot_t* list(slot_t s, int level) noexcept {
    return level == 0 ? links0_.data() + static_cast<std::size_t>(s) * (maxM0_ + 1)
                      : upper_[s].data() + static_cast<std::size_t>(level - 1) * (maxM_ + 1);
  }
  void set_list(slot_t s, int level, const std::vector<Candidate>& nbrs);

  float distance(const float* q, slot_t s, SearchStats* st) const noexcept {
    if (st) ++st->distance_computations;
    return Dist::distance(q, store_.get(s), dim_);
  }
  Candidate make_candidate(float d, slot_t s) const noexcept { return Candidate{d, s, labels_[s]}; }

  Candidate greedy_descend(const float* q, Candidate cur, int from_level, int to_level, SearchStats* st) const;
  std::vector<Candidate> search_layer(const float* q, const std::vector<Candidate>& entry_points, std::size_t ef,
                                      int level, bool exclude_deleted, SearchStats* st) const;
  std::vector<Candidate> select_neighbors(const std::vector<Candidate>& sorted, std::size_t m) const;
  void add_link(slot_t from, slot_t to, int level, float dist);
  void insert_slot(slot_t s, int level);
  void repair_after_delete(slot_t x);
  void choose_new_entry();
  void swap_state(HNSWIndex& other) noexcept;

  std::size_t dim_;
  HnswParams params_;
  std::size_t maxM_;
  std::size_t maxM0_;
  double mult_;
  bool keep_pruned_;  ///< params_.keep_pruned || !Dist::kIsMetric

  VectorStore store_;
  std::vector<label_t> labels_;
  std::vector<std::uint8_t> levels_;
  std::vector<std::uint8_t> deleted_;
  std::vector<slot_t> links0_;
  std::vector<std::vector<slot_t>> upper_;
  OpenHashMap<label_t, slot_t> id_map_;
  slot_t entry_ = kInvalidSlot;
  int max_level_ = -1;
  std::size_t live_ = 0;
  LevelGenerator level_gen_;

  // Search scratch (see thread-safety note above).
  mutable VisitedTable visited_;
  mutable MinHeap<Candidate, CandidateLess> frontier_;
  mutable BoundedMaxHeap<Candidate, CandidateLess> results_;
};

// ======================================================================================
// Implementation
// ======================================================================================

template <typename Dist>
HNSWIndex<Dist>::HNSWIndex(std::size_t dim, const HnswParams& params)
    : dim_(dim),
      params_(params),
      maxM_(params.M),
      maxM0_(params.effective_M0()),
      mult_(1.0 / std::log(static_cast<double>(params.M < 2 ? 2 : params.M))),
      keep_pruned_(params.keep_pruned || !Dist::kIsMetric),
      store_(dim),
      level_gen_(params.seed, 1.0 / std::log(static_cast<double>(params.M < 2 ? 2 : params.M))) {
  params_.validate();
}

template <typename Dist>
void HNSWIndex<Dist>::add(label_t id, const float* v, std::size_t d) {
  validate_vector(v, d);
  if (id_map_.contains(id)) throw InvalidArgument("id " + std::to_string(id) + " already exists");
  if (labels_.size() >= static_cast<std::size_t>(kInvalidSlot) - 1) throw InvalidArgument("index is full");
  std::vector<float> buf(v, v + d);
  Dist::prepare(buf.data(), d);
  add_prepared(id, buf.data());
}

template <typename Dist>
void HNSWIndex<Dist>::add_prepared(label_t id, const float* prepared) {
  const int level = params_.single_layer ? 0 : level_gen_.next();
  const slot_t s = store_.add(prepared);
  labels_.push_back(id);
  levels_.push_back(static_cast<std::uint8_t>(level));
  deleted_.push_back(0);
  links0_.resize(links0_.size() + (maxM0_ + 1), 0);
  upper_.emplace_back(static_cast<std::size_t>(level) * (maxM_ + 1), 0);
  visited_.resize(labels_.size());
  id_map_.insert(id, s);
  ++live_;
  insert_slot(s, level);
  // If every earlier node was deleted the entry point is a tombstone; hand it to a live node.
  if (deleted_[entry_]) choose_new_entry();
}

template <typename Dist>
void HNSWIndex<Dist>::insert_slot(slot_t s, int level) {
  if (entry_ == kInvalidSlot) {
    entry_ = s;
    max_level_ = level;
    return;
  }
  const float* q = store_.get(s);
  Candidate cur = make_candidate(distance(q, entry_, nullptr), entry_);
  cur = greedy_descend(q, cur, max_level_, level, nullptr);
  std::vector<Candidate> eps{cur};
  for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
    std::vector<Candidate> found = search_layer(q, eps, params_.ef_construction, lc, /*exclude_deleted=*/true, nullptr);
    if (found.empty()) {
      // Only tombstones are reachable from here: link to them so the new node stays reachable.
      found = search_layer(q, eps, params_.ef_construction, lc, /*exclude_deleted=*/false, nullptr);
    }
    const std::vector<Candidate> selected = select_neighbors(found, params_.M);
    set_list(s, lc, selected);
    for (const Candidate& c : selected) add_link(c.slot, s, lc, c.dist);
    eps = std::move(found);
  }
  if (level > max_level_) {
    entry_ = s;
    max_level_ = level;
  }
}

template <typename Dist>
typename HNSWIndex<Dist>::Candidate HNSWIndex<Dist>::greedy_descend(const float* q, Candidate cur, int from_level,
                                                                    int to_level, SearchStats* st) const {
  for (int lc = from_level; lc > to_level; --lc) {
    bool changed = true;
    while (changed) {
      changed = false;
      const slot_t* lst = list(cur.slot, lc);
      const std::size_t cnt = lst[0];
      for (std::size_t i = 1; i <= cnt; ++i) {
        const Candidate c = make_candidate(distance(q, lst[i], st), lst[i]);
        if (CandidateLess{}(c, cur)) {
          cur = c;
          changed = true;
        }
      }
    }
  }
  return cur;
}

template <typename Dist>
std::vector<typename HNSWIndex<Dist>::Candidate> HNSWIndex<Dist>::search_layer(
    const float* q, const std::vector<Candidate>& entry_points, std::size_t ef, int level, bool exclude_deleted,
    SearchStats* st) const {
  const CandidateLess less{};
  visited_.reset();
  frontier_.clear();
  results_.reset(ef);
  for (const Candidate& e : entry_points) {
    if (!visited_.visit(e.slot)) continue;
    if (st) ++st->visited;
    frontier_.push(e);
    if (!(exclude_deleted && deleted_[e.slot])) results_.push(e);
  }
  while (!frontier_.empty()) {
    const Candidate c = frontier_.top();
    // Lower-bound stop: the closest unexpanded candidate is worse than the worst kept result.
    if (results_.full() && less(results_.top(), c)) break;
    frontier_.pop();
    if (st) ++st->expanded;
    const slot_t* lst = list(c.slot, level);
    const std::size_t cnt = lst[0];
#if defined(__GNUC__) || defined(__clang__)
    for (std::size_t i = 1; i <= cnt; ++i) __builtin_prefetch(store_.get(lst[i]));
#endif
    for (std::size_t i = 1; i <= cnt; ++i) {
      const slot_t nb = lst[i];
      if (!visited_.visit(nb)) continue;
      if (st) ++st->visited;
      const Candidate cand = make_candidate(distance(q, nb, st), nb);
      if (!results_.full() || less(cand, results_.top())) {
        frontier_.push(cand);
        if (!(exclude_deleted && deleted_[nb])) results_.push(cand);
      }
    }
  }
  return results_.take_sorted();
}

template <typename Dist>
std::vector<typename HNSWIndex<Dist>::Candidate> HNSWIndex<Dist>::select_neighbors(
    const std::vector<Candidate>& sorted, std::size_t m) const {
  if (sorted.size() <= m) return sorted;
  if (params_.selection == NeighborSelection::Simple) {
    return std::vector<Candidate>(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(m));
  }
  std::vector<Candidate> kept;
  std::vector<Candidate> pruned;
  kept.reserve(m);
  for (const Candidate& c : sorted) {
    if (kept.size() >= m) break;
    const float* cv = store_.get(c.slot);
    bool good = true;
    for (const Candidate& r : kept) {
      if (Dist::distance(cv, store_.get(r.slot), dim_) < c.dist) {
        good = false;
        break;
      }
    }
    if (good) {
      kept.push_back(c);
    } else if (keep_pruned_) {
      pruned.push_back(c);
    }
  }
  // `pruned` is already in ascending distance order because `sorted` is.
  for (std::size_t i = 0; i < pruned.size() && kept.size() < m; ++i) kept.push_back(pruned[i]);
  return kept;
}

template <typename Dist>
void HNSWIndex<Dist>::set_list(slot_t s, int level, const std::vector<Candidate>& nbrs) {
  slot_t* lst = list(s, level);
  const std::size_t cnt = std::min(nbrs.size(), max_degree(level));
  lst[0] = static_cast<slot_t>(cnt);
  for (std::size_t i = 0; i < cnt; ++i) lst[i + 1] = nbrs[i].slot;
}

template <typename Dist>
void HNSWIndex<Dist>::add_link(slot_t from, slot_t to, int level, float dist) {
  slot_t* lst = list(from, level);
  const std::size_t cnt = lst[0];
  for (std::size_t i = 1; i <= cnt; ++i) {
    if (lst[i] == to) return;
  }
  if (cnt < max_degree(level)) {
    lst[cnt + 1] = to;
    lst[0] = static_cast<slot_t>(cnt + 1);
    return;
  }
  // Overflow: re-select the neighbourhood of `from` among old neighbours + the new node.
  std::vector<Candidate> cands;
  cands.reserve(cnt + 1);
  const float* base = store_.get(from);
  for (std::size_t i = 1; i <= cnt; ++i) cands.push_back(make_candidate(distance(base, lst[i], nullptr), lst[i]));
  cands.push_back(make_candidate(dist, to));
  std::sort(cands.begin(), cands.end(), CandidateLess{});
  set_list(from, level, select_neighbors(cands, max_degree(level)));
}

template <typename Dist>
std::vector<SearchResult> HNSWIndex<Dist>::search(const float* q, std::size_t d, std::size_t k,
                                                  const SearchParams& params, SearchStats* st) const {
  validate_vector(q, d);
  if (k == 0 || live_ == 0) return {};
  std::vector<float> qbuf(q, q + d);
  Dist::prepare(qbuf.data(), d);
  const float* qp = qbuf.data();
  Candidate ep = make_candidate(distance(qp, entry_, st), entry_);
  ep = greedy_descend(qp, ep, max_level_, 0, st);
  const std::size_t ef = std::max(params.ef, k);
  const std::vector<Candidate> found = search_layer(qp, std::vector<Candidate>{ep}, ef, 0, true, st);
  std::vector<SearchResult> out;
  out.reserve(std::min(k, found.size()));
  for (std::size_t i = 0; i < found.size() && out.size() < k; ++i) {
    out.push_back(SearchResult{found[i].label, found[i].dist});
  }
  return out;
}

template <typename Dist>
void HNSWIndex<Dist>::remove(label_t id) {
  const slot_t* found = id_map_.find(id);
  if (found == nullptr) throw NotFound("id " + std::to_string(id) + " is not in the index");
  const slot_t x = *found;
  id_map_.erase(id);
  deleted_[x] = 1;
  --live_;
  if (params_.delete_policy != DeletePolicy::Tombstone) repair_after_delete(x);
  if (x == entry_) choose_new_entry();
}

// Local repair (in the spirit of FreshDiskANN's delete consolidation): every live out-neighbour u
// of x that links back to x drops that edge and re-selects its neighbourhood among its other
// neighbours plus x's neighbours. The graph is close to symmetric because links are added in both
// directions, so x's out-neighbours approximate its in-neighbours; remaining in-edges keep pointing
// at the tombstone, which stays routable and is never returned, so correctness does not depend on
// finding them (stats() reports how many such edges exist).
template <typename Dist>
void HNSWIndex<Dist>::repair_after_delete(slot_t x) {
  for (int lc = 0; lc <= static_cast<int>(levels_[x]); ++lc) {
    const slot_t* xl = list(x, lc);
    const std::vector<slot_t> xn(xl + 1, xl + 1 + xl[0]);
    for (const slot_t u : xn) {
      if (deleted_[u]) continue;
      slot_t* ul = list(u, lc);
      const std::size_t ucnt = ul[0];
      bool links_x = false;
      for (std::size_t i = 1; i <= ucnt; ++i) links_x = links_x || ul[i] == x;
      if (!links_x) continue;
      const float* uv = store_.get(u);
      std::vector<Candidate> pool;
      pool.reserve(ucnt + xn.size());
      auto consider = [&](slot_t v) {
        if (v == u || v == x || deleted_[v] || static_cast<int>(levels_[v]) < lc) return;
        for (const Candidate& c : pool) {
          if (c.slot == v) return;
        }
        pool.push_back(make_candidate(distance(uv, v, nullptr), v));
      };
      for (std::size_t i = 1; i <= ucnt; ++i) consider(ul[i]);
      for (const slot_t v : xn) consider(v);
      if (pool.empty()) continue;  // only tombstones around u: keep its links for connectivity
      std::sort(pool.begin(), pool.end(), CandidateLess{});
      std::vector<Candidate> kept = select_neighbors(pool, max_degree(lc));
      const DeletePolicy pol = params_.delete_policy;
      if (pol == DeletePolicy::Repair || pol == DeletePolicy::RepairFixed) {
        // Refill: the heuristic's choices first, then the closest pruned candidates, until u has as
        // many links as before (x and any stale tombstone links replaced by live nodes).
        const std::size_t target = std::min(max_degree(lc), ucnt);
        for (std::size_t i = 0; i < pool.size() && kept.size() < target; ++i) {
          bool present = false;
          for (const Candidate& k : kept) present = present || k.slot == pool[i].slot;
          if (!present) kept.push_back(pool[i]);
        }
        // Truncating would drop the farthest kept links -- precisely the long-range links that are
        // often the last in-edge of x's former neighbours -- so only the ablation does it.
        if (pol == DeletePolicy::RepairFixed && kept.size() > target) kept.resize(target);
      }
      set_list(u, lc, kept);
    }
  }
}

template <typename Dist>
void HNSWIndex<Dist>::choose_new_entry() {
  slot_t best = kInvalidSlot;
  int best_level = -1;
  for (slot_t s = 0; s < static_cast<slot_t>(labels_.size()); ++s) {
    if (deleted_[s] || static_cast<int>(levels_[s]) <= best_level) continue;
    best = s;
    best_level = levels_[s];
    if (best_level >= max_level_) break;
  }
  if (best == kInvalidSlot) return;  // nothing live: keep the tombstone as a routing start
  entry_ = best;
  max_level_ = best_level;
}

template <typename Dist>
void HNSWIndex<Dist>::reserve(std::size_t n) {
  store_.reserve(n);
  labels_.reserve(n);
  levels_.reserve(n);
  deleted_.reserve(n);
  links0_.reserve(n * (maxM0_ + 1));
  upper_.reserve(n);
  id_map_.reserve(n);
  visited_.resize(n);
}

template <typename Dist>
void HNSWIndex<Dist>::compact() {
  HNSWIndex fresh(dim_, params_);
  fresh.reserve(live_);
  for (slot_t s = 0; s < static_cast<slot_t>(labels_.size()); ++s) {
    if (!deleted_[s]) fresh.add_prepared(labels_[s], store_.get(s));
  }
  swap_state(fresh);
}

template <typename Dist>
void HNSWIndex<Dist>::swap_state(HNSWIndex& o) noexcept {
  using std::swap;
  swap(dim_, o.dim_);
  swap(params_, o.params_);
  swap(maxM_, o.maxM_);
  swap(maxM0_, o.maxM0_);
  swap(mult_, o.mult_);
  swap(keep_pruned_, o.keep_pruned_);
  swap(store_, o.store_);
  swap(labels_, o.labels_);
  swap(levels_, o.levels_);
  swap(deleted_, o.deleted_);
  swap(links0_, o.links0_);
  swap(upper_, o.upper_);
  swap(id_map_, o.id_map_);
  swap(entry_, o.entry_);
  swap(max_level_, o.max_level_);
  swap(live_, o.live_);
  swap(level_gen_, o.level_gen_);
  swap(visited_, o.visited_);
}

template <typename Dist>
std::vector<label_t> HNSWIndex<Dist>::labels() const {
  std::vector<label_t> out;
  out.reserve(live_);
  for (std::size_t s = 0; s < labels_.size(); ++s) {
    if (!deleted_[s]) out.push_back(labels_[s]);
  }
  return out;
}

template <typename Dist>
bool HNSWIndex<Dist>::get_vector(label_t id, std::vector<float>& out) const {
  const slot_t* s = id_map_.find(id);
  if (s == nullptr) return false;
  const float* p = store_.get(*s);
  out.assign(p, p + dim_);
  return true;
}

template <typename Dist>
int HNSWIndex<Dist>::level_of(label_t id) const {
  const slot_t* s = id_map_.find(id);
  return s == nullptr ? -1 : static_cast<int>(levels_[*s]);
}

template <typename Dist>
std::vector<label_t> HNSWIndex<Dist>::neighbors_of(label_t id, int level) const {
  const slot_t* s = id_map_.find(id);
  if (s == nullptr) throw NotFound("id " + std::to_string(id) + " is not in the index");
  if (level < 0 || level > static_cast<int>(levels_[*s])) return {};
  const slot_t* lst = list(*s, level);
  std::vector<label_t> out;
  for (std::size_t i = 1; i <= lst[0]; ++i) out.push_back(labels_[lst[i]]);
  return out;
}

template <typename Dist>
std::vector<std::pair<label_t, std::size_t>> HNSWIndex<Dist>::layer0_in_degrees() const {
  std::vector<std::size_t> indeg(labels_.size(), 0);
  for (std::size_t s = 0; s < labels_.size(); ++s) {
    if (deleted_[s]) continue;
    const slot_t* lst = list(static_cast<slot_t>(s), 0);
    for (std::size_t i = 1; i <= lst[0]; ++i) ++indeg[lst[i]];
  }
  std::vector<std::pair<label_t, std::size_t>> out;
  out.reserve(live_);
  for (std::size_t s = 0; s < labels_.size(); ++s) {
    if (!deleted_[s]) out.emplace_back(labels_[s], indeg[s]);
  }
  return out;
}

template <typename Dist>
IndexStats HNSWIndex<Dist>::stats() const {
  IndexStats st;
  st.kind = params_.single_layer ? "nsw" : "hnsw";
  st.metric = metric_name(metric());
  st.dim = dim_;
  st.live = live_;
  st.tombstones = labels_.size() - live_;
  st.max_level = max_level_;
  st.has_entry = entry_ != kInvalidSlot;
  if (st.has_entry) {
    st.entry_id = labels_[entry_];
    st.entry_deleted = deleted_[entry_] != 0;
  }
  int top = 0;
  for (std::uint8_t l : levels_) top = std::max(top, static_cast<int>(l));
  st.nodes_per_level.assign(labels_.empty() ? 0 : static_cast<std::size_t>(top) + 1, 0);
  std::vector<std::size_t> live_at(st.nodes_per_level.size(), 0);
  std::vector<std::size_t> deg_sum(st.nodes_per_level.size(), 0);
  for (std::size_t s = 0; s < labels_.size(); ++s) {
    for (int lc = 0; lc <= static_cast<int>(levels_[s]); ++lc) {
      ++st.nodes_per_level[static_cast<std::size_t>(lc)];
      if (deleted_[s]) continue;
      ++live_at[static_cast<std::size_t>(lc)];
      const slot_t* lst = list(static_cast<slot_t>(s), lc);
      deg_sum[static_cast<std::size_t>(lc)] += lst[0];
      st.edges += lst[0];
      for (std::size_t i = 1; i <= lst[0]; ++i) st.edges_to_tombstones += deleted_[lst[i]] ? 1 : 0;
    }
  }
  // Reachability on layer 0 from the entry point through live and deleted nodes alike (searches
  // traverse tombstones): live nodes outside this set can never be returned.
  if (entry_ != kInvalidSlot) {
    std::vector<std::uint8_t> seen(labels_.size(), 0);
    std::vector<slot_t> stack{entry_};
    seen[entry_] = 1;
    while (!stack.empty()) {
      const slot_t s = stack.back();
      stack.pop_back();
      const slot_t* lst = list(s, 0);
      for (std::size_t i = 1; i <= lst[0]; ++i) {
        if (!seen[lst[i]]) {
          seen[lst[i]] = 1;
          stack.push_back(lst[i]);
        }
      }
    }
    for (std::size_t s = 0; s < labels_.size(); ++s) st.unreachable_live += (!deleted_[s] && !seen[s]) ? 1 : 0;
  }
  for (std::size_t l = 0; l < live_at.size(); ++l) {
    st.avg_out_degree.push_back(live_at[l] == 0 ? 0.0
                                                : static_cast<double>(deg_sum[l]) / static_cast<double>(live_at[l]));
  }
  st.vector_bytes = store_.memory_bytes();
  std::size_t graph = links0_.capacity() * sizeof(slot_t) + upper_.capacity() * sizeof(std::vector<slot_t>);
  for (const auto& u : upper_) graph += u.capacity() * sizeof(slot_t);
  st.graph_bytes = graph;
  st.id_map_bytes = id_map_.memory_bytes();
  st.metadata_bytes = labels_.capacity() * sizeof(label_t) + levels_.capacity() + deleted_.capacity() +
                      visited_.memory_bytes();
  st.params = {{"M", std::to_string(params_.M)},
               {"M0", std::to_string(maxM0_)},
               {"ef_construction", std::to_string(params_.ef_construction)},
               {"selection", selection_name(params_.selection)},
               {"delete_policy", delete_policy_name(params_.delete_policy)},
               {"keep_pruned", keep_pruned_ ? "yes" : "no"},
               {"seed", std::to_string(params_.seed)}};
  return st;
}

template <typename Dist>
std::vector<std::string> HNSWIndex<Dist>::check_invariants() const {
  std::vector<std::string> errs;
  auto fail = [&errs](const std::string& m) {
    if (errs.size() < 25) errs.push_back(m);
  };
  const std::size_t n = labels_.size();
  if (store_.size() != n || levels_.size() != n || deleted_.size() != n || upper_.size() != n) {
    fail("per-slot arrays have inconsistent lengths");
    return errs;
  }
  if (links0_.size() != n * (maxM0_ + 1)) {
    fail("layer-0 link array has the wrong length");
    return errs;
  }
  std::size_t live = 0;
  int top = -1;
  for (std::size_t s = 0; s < n; ++s) {
    const std::string who = "slot " + std::to_string(s) + " (id " + std::to_string(labels_[s]) + ")";
    if (deleted_[s] > 1) fail(who + ": deleted flag is not 0/1");
    if (!deleted_[s]) {
      ++live;
      top = std::max(top, static_cast<int>(levels_[s]));
    }
    if (levels_[s] > LevelGenerator::kMaxLevel) {
      fail(who + ": level out of range");
      continue;
    }
    if (params_.single_layer && levels_[s] != 0) fail(who + ": NSW index has a node above layer 0");
    if (upper_[s].size() != static_cast<std::size_t>(levels_[s]) * (maxM_ + 1)) {
      fail(who + ": upper-layer link block has the wrong length");
      continue;
    }
    for (int lc = 0; lc <= static_cast<int>(levels_[s]); ++lc) {
      const slot_t* lst = list(static_cast<slot_t>(s), lc);
      const std::size_t cnt = lst[0];
      if (cnt > max_degree(lc)) {
        fail(who + ": degree " + std::to_string(cnt) + " exceeds bound on layer " + std::to_string(lc));
        continue;
      }
      for (std::size_t i = 1; i <= cnt; ++i) {
        const slot_t nb = lst[i];
        if (nb >= n) {
          fail(who + ": neighbour slot out of range on layer " + std::to_string(lc));
          continue;
        }
        if (nb == s) fail(who + ": self loop on layer " + std::to_string(lc));
        if (static_cast<int>(levels_[nb]) < lc) fail(who + ": neighbour does not exist on layer " + std::to_string(lc));
        for (std::size_t j = i + 1; j <= cnt; ++j) {
          if (lst[j] == nb) fail(who + ": duplicate neighbour on layer " + std::to_string(lc));
        }
      }
    }
  }
  if (live != live_) fail("live counter " + std::to_string(live_) + " != live slots " + std::to_string(live));
  if (id_map_.size() != live_) fail("id map size differs from live count");
  std::string why;
  if (!id_map_.check_invariants(&why)) fail("id map: " + why);
  id_map_.for_each([&](const label_t& id, const slot_t& s) {
    if (s >= n || labels_[s] != id || deleted_[s]) fail("id map entry for id " + std::to_string(id) + " is stale");
  });
  if (n == 0) {
    if (entry_ != kInvalidSlot || max_level_ != -1) fail("empty index has an entry point");
  } else {
    if (entry_ >= n) {
      fail("entry point out of range");
    } else {
      if (max_level_ != static_cast<int>(levels_[entry_])) fail("max level differs from the entry point's level");
      if (live_ > 0 && deleted_[entry_]) fail("entry point is deleted although live nodes exist");
      if (live_ > 0 && top > max_level_) fail("a live node sits above the entry point's level");
    }
  }
  return errs;
}

template <typename Dist>
std::string HNSWIndex<Dist>::describe() const {
  std::string s = params_.single_layer ? "nsw(" : "hnsw(";
  s += std::string(metric_name(metric())) + ", d=" + std::to_string(dim_) + ", M=" + std::to_string(params_.M) +
       ", M0=" + std::to_string(maxM0_) + ", efc=" + std::to_string(params_.ef_construction) + ", " +
       selection_name(params_.selection) + (keep_pruned_ ? "+keep-pruned" : "") + ", " +
       delete_policy_name(params_.delete_policy) + ")";
  return s;
}

template <typename Dist>
void HNSWIndex<Dist>::save_body(BinaryWriter& w) const {
  w.write_pod<std::uint64_t>(params_.M);
  w.write_pod<std::uint64_t>(maxM0_);
  w.write_pod<std::uint64_t>(params_.ef_construction);
  w.write_pod<std::uint64_t>(params_.seed);
  w.write_pod<std::uint8_t>(static_cast<std::uint8_t>(params_.selection));
  w.write_pod<std::uint8_t>(static_cast<std::uint8_t>(params_.delete_policy));
  w.write_pod<std::uint8_t>(params_.single_layer ? 1 : 0);
  w.write_pod<std::uint8_t>(params_.keep_pruned ? 1 : 0);
  w.write_pod<std::uint64_t>(labels_.size());
  w.write_pod<std::uint64_t>(live_);
  w.write_pod<std::uint32_t>(entry_);
  w.write_pod<std::int32_t>(max_level_);
  w.write_string(level_gen_.state());
  w.write_array(store_.raw());
  w.write_array(labels_);
  w.write_array(levels_);
  w.write_array(deleted_);
  w.write_array(links0_);
  for (const auto& u : upper_) w.write_array(u);
}

template <typename Dist>
std::unique_ptr<HNSWIndex<Dist>> HNSWIndex<Dist>::load_body(BinaryReader& r, std::size_t dim) {
  HnswParams p;
  p.M = static_cast<std::size_t>(r.read_u64_in_range("M", 2, 4096));
  p.M0 = static_cast<std::size_t>(r.read_u64_in_range("M0", 2, 8192));
  p.ef_construction = static_cast<std::size_t>(r.read_u64_in_range("ef_construction", 1, 1u << 20));
  p.seed = r.read_pod<std::uint64_t>();
  const auto sel = r.read_pod<std::uint8_t>();
  const auto pol = r.read_pod<std::uint8_t>();
  const auto single = r.read_pod<std::uint8_t>();
  const auto keep = r.read_pod<std::uint8_t>();
  if (sel > 1 || pol > 3 || single > 1 || keep > 1) throw FormatError("index file has an invalid HNSW option byte");
  p.selection = static_cast<NeighborSelection>(sel);
  p.delete_policy = static_cast<DeletePolicy>(pol);
  p.single_layer = single != 0;
  p.keep_pruned = keep != 0;
  try {
    p.validate();
  } catch (const InvalidArgument& e) {
    throw FormatError(std::string("index file has invalid HNSW parameters: ") + e.what());
  }
  auto idx = std::make_unique<HNSWIndex>(dim, p);
  const std::uint64_t n = r.read_u64_in_range("slot count", 0, kInvalidSlot - 1);
  const std::uint64_t live = r.read_u64_in_range("live count", 0, n);
  const auto entry = r.read_pod<std::uint32_t>();
  const auto max_level = r.read_pod<std::int32_t>();
  idx->level_gen_.restore(r.read_string(1u << 16));
  std::vector<float> raw = r.read_array<float>(n * dim);
  for (float x : raw) {
    if (!std::isfinite(x)) throw FormatError("index file contains a non-finite vector component");
  }
  idx->store_ = VectorStore::from_raw(dim, std::move(raw));
  idx->labels_ = r.read_array<label_t>(n);
  idx->levels_ = r.read_array<std::uint8_t>(n);
  idx->deleted_ = r.read_array<std::uint8_t>(n);
  idx->links0_ = r.read_array<slot_t>(n * (idx->maxM0_ + 1));
  idx->upper_.resize(static_cast<std::size_t>(n));
  for (std::size_t s = 0; s < n; ++s) {
    if (idx->levels_[s] > LevelGenerator::kMaxLevel) throw FormatError("index file has a node level out of range");
    idx->upper_[s] = r.read_array<slot_t>(static_cast<std::uint64_t>(idx->levels_[s]) * (idx->maxM_ + 1));
  }
  idx->id_map_.reserve(static_cast<std::size_t>(live));
  for (std::size_t s = 0; s < n; ++s) {
    if (idx->deleted_[s]) continue;
    if (!idx->id_map_.insert(idx->labels_[s], static_cast<slot_t>(s))) {
      throw FormatError("index file contains duplicate live id " + std::to_string(idx->labels_[s]));
    }
  }
  idx->live_ = static_cast<std::size_t>(live);
  idx->entry_ = entry;
  idx->max_level_ = max_level;
  idx->visited_.resize(static_cast<std::size_t>(n));
  const std::vector<std::string> problems = idx->check_invariants();
  if (!problems.empty()) throw FormatError("index file failed the consistency check: " + problems.front());
  return idx;
}

extern template class HNSWIndex<L2>;
extern template class HNSWIndex<InnerProduct>;
extern template class HNSWIndex<Cosine>;

}  // namespace hnsw
