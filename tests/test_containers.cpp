// Unit tests for the custom data structures: MinHeap, BoundedMaxHeap, OpenHashMap, quickselect,
// VisitedTable and VectorStore. Each is checked against an STL reference under random operation
// sequences, with its structural invariant verified after every operation.
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <unordered_map>

#include "hnsw/hash_table.h"
#include "hnsw/heap.h"
#include "hnsw/top_k.h"
#include "hnsw/vector_store.h"
#include "hnsw/visited_table.h"
#include "test_framework.h"

using namespace hnsw;

TEST(min_heap_matches_multiset_under_random_ops) {
  std::mt19937 rng(1);
  MinHeap<int> heap;
  std::multiset<int> ref;
  for (int step = 0; step < 20000; ++step) {
    if (ref.empty() || rng() % 3 != 0) {
      const int v = static_cast<int>(rng() % 1000);  // many duplicates
      heap.push(v);
      ref.insert(v);
    } else {
      REQUIRE(!heap.empty());
      CHECK_EQ(heap.top(), *ref.begin());
      heap.pop();
      ref.erase(ref.begin());
    }
    CHECK_EQ(heap.size(), ref.size());
    if (step % 97 == 0) CHECK(heap.is_valid());
  }
  while (!heap.empty()) {
    CHECK_EQ(heap.extract_top(), *ref.begin());
    ref.erase(ref.begin());
  }
  CHECK(ref.empty());
}

TEST(min_heap_custom_comparator_and_move_only_type) {
  // std::greater turns it into a max-heap; unique_ptr checks that no copies are required.
  MinHeap<int, std::greater<int>> maxheap;
  for (int v : {5, 1, 9, 3, 7}) maxheap.push(v);
  CHECK_EQ(maxheap.top(), 9);
  auto cmp = [](const std::unique_ptr<int>& a, const std::unique_ptr<int>& b) { return *a < *b; };
  MinHeap<std::unique_ptr<int>, decltype(cmp)> h(cmp);
  for (int v : {4, 2, 8, 6}) h.push(std::make_unique<int>(v));
  std::vector<int> out;
  while (!h.empty()) out.push_back(*h.extract_top());
  CHECK(out == std::vector<int>({2, 4, 6, 8}));
}

TEST(min_heap_single_element_and_clear) {
  MinHeap<double> h;
  h.push(3.5);
  CHECK_EQ(h.top(), 3.5);
  h.pop();
  CHECK(h.empty());
  h.push(1.0);
  h.push(2.0);
  h.clear();
  CHECK(h.empty());
  CHECK(h.is_valid());
}

TEST(bounded_max_heap_keeps_k_smallest) {
  std::mt19937 rng(2);
  for (std::size_t cap : {0u, 1u, 2u, 7u, 64u}) {
    BoundedMaxHeap<int> h(cap);
    std::vector<int> all;
    for (int i = 0; i < 500; ++i) {
      const int v = static_cast<int>(rng() % 200);
      all.push_back(v);
      const bool expect = h.accepts(v);
      CHECK_EQ(h.push(v), expect);
      CHECK(h.is_valid());
      CHECK(h.size() <= cap);
    }
    std::sort(all.begin(), all.end());
    all.resize(std::min<std::size_t>(cap, all.size()));
    CHECK(h.take_sorted() == all);
    CHECK(h.empty());
  }
}

TEST(bounded_max_heap_threshold_and_reset) {
  BoundedMaxHeap<int> h(3);
  for (int v : {10, 20, 30}) h.push(v);
  CHECK(h.full());
  CHECK_EQ(h.top(), 30);
  CHECK(!h.accepts(30));  // equal to the threshold is not strictly better
  CHECK(h.accepts(29));
  CHECK(h.push(5));
  CHECK_EQ(h.top(), 20);
  h.reset(1);
  CHECK(h.empty());
  CHECK_EQ(h.capacity(), 1u);
  h.push(4);
  h.push(2);
  CHECK_EQ(h.top(), 2);
}

TEST(hash_map_matches_unordered_map_under_random_ops) {
  std::mt19937_64 rng(3);
  OpenHashMap<std::uint64_t, std::uint32_t> map(4);
  std::unordered_map<std::uint64_t, std::uint32_t> ref;
  for (int step = 0; step < 50000; ++step) {
    const std::uint64_t key = rng() % 3000;
    const int op = static_cast<int>(rng() % 4);
    if (op == 0 || op == 1) {
      const auto val = static_cast<std::uint32_t>(rng());
      const bool inserted = map.insert(key, val);
      const bool ref_inserted = ref.emplace(key, val).second;
      CHECK_EQ(inserted, ref_inserted);
    } else if (op == 2) {
      CHECK_EQ(map.erase(key), ref.erase(key) == 1);
    } else {
      const std::uint32_t* v = map.find(key);
      const auto it = ref.find(key);
      CHECK_EQ(v != nullptr, it != ref.end());
      if (v != nullptr && it != ref.end()) CHECK_EQ(*v, it->second);
    }
    CHECK_EQ(map.size(), ref.size());
    if (step % 1000 == 0) {
      std::string why;
      CHECK(map.check_invariants(&why));
    }
  }
  std::string why;
  CHECK(map.check_invariants(&why));
  std::size_t visited = 0;
  map.for_each([&](const std::uint64_t& k, const std::uint32_t& v) {
    ++visited;
    CHECK(ref.count(k) == 1 && ref[k] == v);
  });
  CHECK_EQ(visited, ref.size());
}

/// A deliberately terrible hash that sends every key to the same bucket.
struct ConstantHash {
  std::size_t operator()(std::uint64_t) const noexcept { return 5; }
};

TEST(hash_map_full_collisions_and_backward_shift_delete) {
  OpenHashMap<std::uint64_t, int, ConstantHash> map(16);
  for (std::uint64_t k = 0; k < 40; ++k) CHECK(map.insert(k, static_cast<int>(k) * 10));
  std::string why;
  CHECK(map.check_invariants(&why));
  CHECK_EQ(map.probe_length(0), 1u);
  CHECK_EQ(map.probe_length(39), 40u);
  // Delete from the middle of the single probe chain; every remaining key must stay reachable.
  for (std::uint64_t k = 0; k < 40; k += 3) CHECK(map.erase(k));
  CHECK(map.check_invariants(&why));
  for (std::uint64_t k = 0; k < 40; ++k) {
    const int* v = map.find(k);
    if (k % 3 == 0) {
      CHECK(v == nullptr);
    } else {
      REQUIRE(v != nullptr);
      CHECK_EQ(*v, static_cast<int>(k) * 10);
    }
  }
  CHECK(!map.erase(1000));
  CHECK(!map.insert(1, 0));  // duplicate keeps the old value
  CHECK_EQ(*map.find(1), 10);
  map.insert_or_assign(1, 11);
  CHECK_EQ(*map.find(1), 11);
}

TEST(hash_map_growth_reserve_and_clear) {
  OpenHashMap<std::uint64_t, std::uint64_t> map;
  map.reserve(1000);
  const std::size_t cap = map.capacity();
  for (std::uint64_t k = 0; k < 1000; ++k) map.insert(k * 7919, k);
  CHECK_EQ(map.capacity(), cap);  // reserve avoided rehashing
  CHECK(map.load_factor() <= 0.7);
  for (std::uint64_t k = 0; k < 1000; ++k) CHECK_EQ(*map.find(k * 7919), k);
  map.clear();
  CHECK(map.empty());
  CHECK(!map.contains(0));
  std::string why;
  CHECK(map.check_invariants(&why));
}

TEST(quickselect_matches_sort_on_adversarial_inputs) {
  std::mt19937 rng(4);
  auto run = [](std::vector<int> v, std::size_t k) {
    std::vector<int> expect = v;
    std::sort(expect.begin(), expect.end());
    expect.resize(std::min(k, expect.size()));
    select_smallest(v.begin(), v.size(), k, std::less<int>());
    v.resize(std::min(k, v.size()));
    std::sort(v.begin(), v.end());
    return v == expect;
  };
  for (std::size_t n : {0u, 1u, 2u, 15u, 16u, 17u, 100u, 5000u}) {
    std::vector<int> random(n), sorted(n), reversed(n), equal(n, 7), few(n);
    for (std::size_t i = 0; i < n; ++i) {
      random[i] = static_cast<int>(rng() % 100000);
      sorted[i] = static_cast<int>(i);
      reversed[i] = static_cast<int>(n - i);
      few[i] = static_cast<int>(rng() % 3);
    }
    for (std::size_t k : {std::size_t{0}, std::size_t{1}, n / 2, n == 0 ? 0 : n - 1, n, n + 5}) {
      CHECK(run(random, k));
      CHECK(run(sorted, k));
      CHECK(run(reversed, k));
      CHECK(run(equal, k));
      CHECK(run(few, k));
    }
  }
}

TEST(insertion_sort_sorts) {
  std::vector<int> v = {5, 3, 9, 1, 1, 0, 7};
  insertion_sort(v.begin(), v.end(), std::less<int>());
  CHECK(std::is_sorted(v.begin(), v.end()));
}

TEST(visited_table_epochs_and_wraparound) {
  VisitedTable vt(10);
  vt.reset();
  CHECK(vt.visit(3));
  CHECK(!vt.visit(3));
  CHECK(vt.visited(3));
  CHECK(!vt.visited(4));
  // Force the 16-bit epoch to wrap: marks must never leak between searches.
  for (int i = 0; i < 70000; ++i) {
    vt.reset();
    CHECK(vt.visit(static_cast<slot_t>(i % 10)));
    if (!vt.visited(static_cast<slot_t>(i % 10))) break;
  }
  vt.reset();
  for (slot_t s = 0; s < 10; ++s) CHECK(!vt.visited(s));
  vt.resize(20);
  CHECK_EQ(vt.size(), 20u);
  CHECK(vt.visit(19));
}

TEST(vector_store_add_set_pop_and_self_alias) {
  VectorStore s(3);
  const float a[3] = {1, 2, 3};
  const float b[3] = {4, 5, 6};
  CHECK_EQ(s.add(a), 0u);
  CHECK_EQ(s.add(b), 1u);
  // Adding a vector that lives inside the store itself must survive reallocation.
  for (int i = 0; i < 100; ++i) s.add(s.get(0));
  CHECK_EQ(s.size(), 102u);
  CHECK_EQ(s.get(101)[2], 3.0f);
  s.set(0, b);
  CHECK_EQ(s.get(0)[0], 4.0f);
  s.pop_back();
  CHECK_EQ(s.size(), 101u);
  CHECK_THROWS_AS(VectorStore(0), InvalidArgument);
  CHECK_THROWS_AS(VectorStore::from_raw(3, std::vector<float>(4)), FormatError);
  const VectorStore r = VectorStore::from_raw(2, {1, 2, 3, 4});
  CHECK_EQ(r.size(), 2u);
  CHECK_EQ(r.get(1)[1], 4.0f);
}
