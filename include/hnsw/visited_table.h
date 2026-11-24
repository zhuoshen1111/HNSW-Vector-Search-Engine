// VisitedTable: O(1) "have I seen this slot during the current search?" set.
//
// Each slot stores the epoch in which it was last visited. reset() starts a new search by
// incrementing the epoch, so clearing costs O(1) instead of O(N); the array is only wiped when
// the 16-bit epoch counter wraps (once every 65535 searches).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hnsw/common.h"

namespace hnsw {

class VisitedTable {
 public:
  explicit VisitedTable(std::size_t n = 0) : marks_(n, 0) {}

  /// Grows the table to cover slots [0, n). Never shrinks.
  void resize(std::size_t n) {
    if (n > marks_.size()) marks_.resize(n, 0);
  }

  /// Starts a new search.
  void reset() noexcept {
    ++epoch_;
    if (epoch_ == 0) {
      std::fill(marks_.begin(), marks_.end(), std::uint16_t{0});
      epoch_ = 1;
    }
  }

  /// Marks s as visited; returns true if it had not been visited in this epoch.
  bool visit(slot_t s) noexcept {
    if (marks_[s] == epoch_) return false;
    marks_[s] = epoch_;
    return true;
  }

  bool visited(slot_t s) const noexcept { return marks_[s] == epoch_; }
  std::size_t size() const noexcept { return marks_.size(); }
  std::size_t memory_bytes() const noexcept { return marks_.capacity() * sizeof(std::uint16_t); }

 private:
  std::vector<std::uint16_t> marks_;
  std::uint16_t epoch_ = 1;
};

}  // namespace hnsw
