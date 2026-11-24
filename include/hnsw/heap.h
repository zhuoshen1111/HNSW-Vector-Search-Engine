// Custom binary heaps used by graph search and exact top-k selection.
//
//   MinHeap<T, Compare>         unbounded priority queue whose top() is the minimum under Compare
//                               (the HNSW candidate frontier).
//   BoundedMaxHeap<T, Compare>  keeps the `capacity` smallest elements seen so far; top() is the
//                               largest kept element, i.e. the current admission threshold
//                               (the HNSW result set W and streaming top-k).
//
// Both store elements in a std::vector laid out as an implicit binary tree; sift-up/sift-down
// are implemented here (std::priority_queue / std::push_heap are not used).
#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace hnsw {
namespace detail {

/// `higher(a, b)` returns true when `a` must sit closer to the root than `b`.
template <typename T, typename Higher>
void sift_up(std::vector<T>& h, std::size_t i, const Higher& higher) {
  T item = std::move(h[i]);
  while (i > 0) {
    const std::size_t parent = (i - 1) / 2;
    if (!higher(item, h[parent])) break;
    h[i] = std::move(h[parent]);
    i = parent;
  }
  h[i] = std::move(item);
}

template <typename T, typename Higher>
void sift_down(std::vector<T>& h, std::size_t i, const Higher& higher) {
  const std::size_t n = h.size();
  T item = std::move(h[i]);
  for (;;) {
    std::size_t child = 2 * i + 1;
    if (child >= n) break;
    if (child + 1 < n && higher(h[child + 1], h[child])) ++child;
    if (!higher(h[child], item)) break;
    h[i] = std::move(h[child]);
    i = child;
  }
  h[i] = std::move(item);
}

template <typename T, typename Higher>
bool is_heap_ordered(const std::vector<T>& h, const Higher& higher) {
  for (std::size_t i = 1; i < h.size(); ++i) {
    if (higher(h[i], h[(i - 1) / 2])) return false;
  }
  return true;
}

}  // namespace detail

template <typename T, typename Compare = std::less<T>>
class MinHeap {
 public:
  MinHeap() = default;
  explicit MinHeap(Compare comp) : comp_(std::move(comp)) {}

  void push(const T& value) {
    data_.push_back(value);
    detail::sift_up(data_, data_.size() - 1, higher());
  }
  void push(T&& value) {
    data_.push_back(std::move(value));
    detail::sift_up(data_, data_.size() - 1, higher());
  }
  template <typename... Args>
  void emplace(Args&&... args) {
    data_.emplace_back(std::forward<Args>(args)...);
    detail::sift_up(data_, data_.size() - 1, higher());
  }

  /// Precondition: !empty().
  const T& top() const {
    assert(!data_.empty());
    return data_.front();
  }
  /// Removes the minimum. Precondition: !empty().
  void pop() {
    assert(!data_.empty());
    if (data_.size() == 1) {
      data_.pop_back();
      return;
    }
    data_.front() = std::move(data_.back());
    data_.pop_back();
    detail::sift_down(data_, 0, higher());
  }
  /// Removes and returns the minimum. Precondition: !empty().
  T extract_top() {
    T out = std::move(data_.front());
    if (data_.size() == 1) {
      data_.pop_back();
      return out;
    }
    data_.front() = std::move(data_.back());
    data_.pop_back();
    detail::sift_down(data_, 0, higher());
    return out;
  }

  bool empty() const noexcept { return data_.empty(); }
  std::size_t size() const noexcept { return data_.size(); }
  void clear() noexcept { data_.clear(); }
  void reserve(std::size_t n) { data_.reserve(n); }

  /// Heap-order invariant: no child compares smaller than its parent.
  bool is_valid() const { return detail::is_heap_ordered(data_, higher()); }
  /// Underlying storage in heap (not sorted) order.
  const std::vector<T>& unordered() const noexcept { return data_; }

 private:
  auto higher() const {
    return [this](const T& a, const T& b) { return comp_(a, b); };
  }
  std::vector<T> data_;
  Compare comp_{};
};

template <typename T, typename Compare = std::less<T>>
class BoundedMaxHeap {
 public:
  explicit BoundedMaxHeap(std::size_t capacity = 0, Compare comp = Compare())
      : capacity_(capacity), comp_(std::move(comp)) {
    data_.reserve(capacity);
  }

  std::size_t capacity() const noexcept { return capacity_; }
  std::size_t size() const noexcept { return data_.size(); }
  bool empty() const noexcept { return data_.empty(); }
  bool full() const noexcept { return data_.size() >= capacity_; }

  /// True iff push(value) would keep the element.
  bool accepts(const T& value) const {
    if (capacity_ == 0) return false;
    return !full() || comp_(value, data_.front());
  }

  /// Offers an element; keeps it if it is among the `capacity` smallest. Returns true if kept.
  bool push(const T& value) {
    if (capacity_ == 0) return false;
    if (!full()) {
      data_.push_back(value);
      detail::sift_up(data_, data_.size() - 1, higher());
      return true;
    }
    if (!comp_(value, data_.front())) return false;
    data_.front() = value;
    detail::sift_down(data_, 0, higher());
    return true;
  }

  /// Largest kept element (the admission threshold). Precondition: !empty().
  const T& top() const {
    assert(!data_.empty());
    return data_.front();
  }
  /// Removes the largest kept element. Precondition: !empty().
  void pop() {
    assert(!data_.empty());
    if (data_.size() == 1) {
      data_.pop_back();
      return;
    }
    data_.front() = std::move(data_.back());
    data_.pop_back();
    detail::sift_down(data_, 0, higher());
  }

  void clear() noexcept { data_.clear(); }
  /// Clears and changes the capacity.
  void reset(std::size_t capacity) {
    data_.clear();
    capacity_ = capacity;
    data_.reserve(capacity);
  }

  /// Heap sort: returns the kept elements in ascending order and leaves the heap empty.
  std::vector<T> take_sorted() {
    std::vector<T> out(data_.size());
    for (std::size_t i = out.size(); i-- > 0;) {
      out[i] = data_.front();
      pop();
    }
    return out;
  }

  bool is_valid() const { return data_.size() <= capacity_ && detail::is_heap_ordered(data_, higher()); }
  const std::vector<T>& unordered() const noexcept { return data_; }

 private:
  // Max-heap: `a` is above `b` when b < a.
  auto higher() const {
    return [this](const T& a, const T& b) { return comp_(b, a); };
  }
  std::size_t capacity_;
  std::vector<T> data_;
  Compare comp_;
};

}  // namespace hnsw
