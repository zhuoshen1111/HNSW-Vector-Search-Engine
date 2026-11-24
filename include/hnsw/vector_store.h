// VectorStore: owns N fixed-dimension float vectors in one contiguous, row-major buffer.
//
// Slot s occupies floats [s*dim, (s+1)*dim). Contiguity keeps distance computations
// cache-friendly and makes serialisation a single block write. The store owns its memory
// through std::vector (RAII); copying is allowed (deep copy), moving is cheap.
#pragma once

#include <cstddef>
#include <vector>

#include "hnsw/common.h"

namespace hnsw {

class VectorStore {
 public:
  /// Throws InvalidArgument if dim is 0 or larger than kMaxDim.
  explicit VectorStore(std::size_t dim);

  std::size_t dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return count_; }

  /// Appends a copy of v (dim floats) and returns its slot. Safe even if v points into this store.
  slot_t add(const float* v);
  /// Overwrites slot s with v. Precondition: s < size().
  void set(slot_t s, const float* v);
  /// Removes the last vector.
  void pop_back();

  const float* get(slot_t s) const noexcept { return data_.data() + static_cast<std::size_t>(s) * dim_; }

  void reserve(std::size_t n) { data_.reserve(n * dim_); }
  void clear() noexcept {
    data_.clear();
    count_ = 0;
  }

  /// Bytes reserved for vector payloads (capacity, not just size).
  std::size_t memory_bytes() const noexcept { return data_.capacity() * sizeof(float); }

  /// Raw row-major payload, used by the serializer.
  const std::vector<float>& raw() const noexcept { return data_; }
  /// Rebuilds a store from a raw payload. Throws FormatError if the size is not a multiple of dim.
  static VectorStore from_raw(std::size_t dim, std::vector<float> data);

 private:
  std::size_t dim_;
  std::size_t count_ = 0;
  std::vector<float> data_;
};

}  // namespace hnsw
