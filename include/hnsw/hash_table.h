// OpenHashMap<Key, Value, Hash>: open addressing with linear probing and backward-shift deletion.
//
// Used as the external-id -> slot map of both indexes. Properties:
//   * power-of-two capacity, maximum load factor 0.7, doubling growth (amortised O(1) insert);
//   * expected O(1) find/insert/erase under a good hash; O(n) worst case (all keys collide);
//   * erase uses backward shifting instead of tombstones, so probe sequences never degrade
//     after many deletions, which matters for delete-heavy workloads.
// Invariant (checked by check_invariants): for every occupied position p with home bucket h,
// all positions h, h+1, ..., p (cyclically) are occupied.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hnsw {

/// SplitMix64 finaliser: a cheap, well-mixing hash for integer keys.
struct IntegerHash {
  std::size_t operator()(std::uint64_t x) const noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<std::size_t>(x);
  }
};

template <typename Key, typename Value, typename Hash = IntegerHash>
class OpenHashMap {
 public:
  explicit OpenHashMap(std::size_t initial_capacity = 16, Hash hash = Hash()) : hasher_(std::move(hash)) {
    allocate(round_up_pow2(initial_capacity < 8 ? 8 : initial_capacity));
  }

  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  std::size_t capacity() const noexcept { return used_.size(); }
  double load_factor() const noexcept {
    return static_cast<double>(size_) / static_cast<double>(capacity());
  }

  /// Inserts (key, value) if key is absent. Returns false (and changes nothing) if present.
  bool insert(const Key& key, const Value& value) {
    if (locate(key) != npos) return false;
    grow_if_needed();
    place(key, value);
    return true;
  }

  void insert_or_assign(const Key& key, const Value& value) {
    const std::size_t pos = locate(key);
    if (pos != npos) {
      values_[pos] = value;
      return;
    }
    grow_if_needed();
    place(key, value);
  }

  Value* find(const Key& key) noexcept {
    const std::size_t pos = locate(key);
    return pos == npos ? nullptr : &values_[pos];
  }
  const Value* find(const Key& key) const noexcept {
    const std::size_t pos = locate(key);
    return pos == npos ? nullptr : &values_[pos];
  }
  bool contains(const Key& key) const noexcept { return locate(key) != npos; }

  /// Removes key if present (backward-shift deletion). Returns whether it was present.
  bool erase(const Key& key) {
    std::size_t hole = locate(key);
    if (hole == npos) return false;
    used_[hole] = 0;
    --size_;
    std::size_t j = hole;
    for (;;) {
      j = (j + 1) & mask_;
      if (!used_[j]) break;
      const std::size_t h = home(keys_[j]);
      // Entry j may stay only if its home lies cyclically in (hole, j].
      const bool stays = (hole <= j) ? (hole < h && h <= j) : (hole < h || h <= j);
      if (!stays) {
        keys_[hole] = std::move(keys_[j]);
        values_[hole] = std::move(values_[j]);
        used_[hole] = 1;
        used_[j] = 0;
        hole = j;
      }
    }
    return true;
  }

  void clear() {
    std::fill(used_.begin(), used_.end(), std::uint8_t{0});
    size_ = 0;
  }

  /// Ensures n elements fit without rehashing.
  void reserve(std::size_t n) {
    std::size_t cap = capacity();
    while (n * kLoadDen > cap * kLoadNum) cap *= 2;
    if (cap != capacity()) rehash(cap);
  }

  template <typename F>
  void for_each(F&& f) const {
    for (std::size_t i = 0; i < used_.size(); ++i) {
      if (used_[i]) f(keys_[i], values_[i]);
    }
  }

  /// Number of probes needed to find key (1 = found in its home bucket); 0 if absent.
  std::size_t probe_length(const Key& key) const noexcept {
    const std::size_t pos = locate(key);
    if (pos == npos) return 0;
    return ((pos - home(key)) & mask_) + 1;
  }

  std::size_t memory_bytes() const noexcept {
    return keys_.capacity() * sizeof(Key) + values_.capacity() * sizeof(Value) + used_.capacity();
  }

  /// Verifies the linear-probing invariant, the size counter and the load bound.
  bool check_invariants(std::string* why = nullptr) const {
    auto fail = [why](const std::string& msg) {
      if (why) *why = msg;
      return false;
    };
    if (capacity() == 0 || (capacity() & mask_) != 0 || mask_ + 1 != capacity()) return fail("capacity is not a power of two");
    std::size_t count = 0;
    for (std::size_t p = 0; p < used_.size(); ++p) {
      if (!used_[p]) continue;
      ++count;
      const std::size_t h = home(keys_[p]);
      for (std::size_t q = h; q != p; q = (q + 1) & mask_) {
        if (!used_[q]) return fail("probe chain broken: empty bucket between home and entry");
      }
      for (std::size_t q = (p + 1) & mask_; q != p && used_[q]; q = (q + 1) & mask_) {
        if (keys_[q] == keys_[p]) return fail("duplicate key in table");
      }
    }
    if (count != size_) return fail("size counter does not match occupied buckets");
    if (size_ * kLoadDen > capacity() * kLoadNum) return fail("load factor above maximum");
    return true;
  }

 private:
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);
  static constexpr std::size_t kLoadNum = 7;
  static constexpr std::size_t kLoadDen = 10;

  static std::size_t round_up_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  std::size_t home(const Key& key) const noexcept { return hasher_(key) & mask_; }

  std::size_t locate(const Key& key) const noexcept {
    std::size_t i = home(key);
    while (used_[i]) {
      if (keys_[i] == key) return i;
      i = (i + 1) & mask_;
    }
    return npos;
  }

  void place(const Key& key, const Value& value) {
    std::size_t i = home(key);
    while (used_[i]) i = (i + 1) & mask_;
    keys_[i] = key;
    values_[i] = value;
    used_[i] = 1;
    ++size_;
  }

  void grow_if_needed() {
    if ((size_ + 1) * kLoadDen > capacity() * kLoadNum) rehash(capacity() * 2);
  }

  void allocate(std::size_t cap) {
    keys_.assign(cap, Key{});
    values_.assign(cap, Value{});
    used_.assign(cap, 0);
    mask_ = cap - 1;
    size_ = 0;
  }

  void rehash(std::size_t new_cap) {
    std::vector<Key> old_keys = std::move(keys_);
    std::vector<Value> old_values = std::move(values_);
    std::vector<std::uint8_t> old_used = std::move(used_);
    allocate(new_cap);
    for (std::size_t i = 0; i < old_used.size(); ++i) {
      if (old_used[i]) place(old_keys[i], old_values[i]);
    }
  }

  std::vector<Key> keys_;
  std::vector<Value> values_;
  std::vector<std::uint8_t> used_;
  std::size_t size_ = 0;
  std::size_t mask_ = 0;
  Hash hasher_;
};

}  // namespace hnsw
