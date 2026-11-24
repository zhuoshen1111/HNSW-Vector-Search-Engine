// Exact top-k selection (custom quickselect) used by the brute-force reference index.
#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

namespace hnsw {

/// Sorts [first, last) by insertion sort. O(n^2); used for short ranges only.
template <typename It, typename Less>
void insertion_sort(It first, It last, Less less) {
  if (first == last) return;
  for (It i = std::next(first); i != last; ++i) {
    auto value = std::move(*i);
    It j = i;
    while (j != first && less(value, *std::prev(j))) {
      *j = std::move(*std::prev(j));
      --j;
    }
    *j = std::move(value);
  }
}

/// Rearranges [first, first + n) so that the k smallest elements under `less` occupy the first
/// k positions (in unspecified order). Hoare-partition quickselect with median-of-three pivots:
/// expected O(n), worst case O(n^2) (not reachable by sorted, reversed or constant inputs thanks
/// to the median-of-three pivot). Ranges of <= 16 elements are finished by insertion sort.
template <typename It, typename Less>
void select_smallest(It first, std::size_t n, std::size_t k, Less less) {
  using std::swap;
  if (k == 0 || k >= n) return;
  using diff_t = typename std::iterator_traits<It>::difference_type;
  const diff_t nth = static_cast<diff_t>(k);  // after the call, position nth holds the (k+1)-th smallest
  diff_t lo = 0;
  diff_t hi = static_cast<diff_t>(n) - 1;
  while (hi - lo > 16) {
    const diff_t mid = lo + (hi - lo) / 2;
    if (less(first[mid], first[lo])) swap(first[mid], first[lo]);
    if (less(first[hi], first[lo])) swap(first[hi], first[lo]);
    if (less(first[hi], first[mid])) swap(first[hi], first[mid]);
    const auto pivot = first[mid];
    diff_t i = lo - 1;
    diff_t j = hi + 1;
    for (;;) {
      do {
        ++i;
      } while (less(first[i], pivot));
      do {
        --j;
      } while (less(pivot, first[j]));
      if (i >= j) break;
      swap(first[i], first[j]);
    }
    // Now every element of [lo, j] is <= pivot and every element of [j+1, hi] is >= pivot.
    if (nth <= j) {
      hi = j;
    } else {
      lo = j + 1;
    }
  }
  insertion_sort(first + lo, first + hi + 1, less);
}

}  // namespace hnsw
