#include "hnsw/vector_store.h"

#include <algorithm>
#include <string>

namespace hnsw {

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
  if (dim == 0 || dim > kMaxDim) {
    throw InvalidArgument("vector dimension must be in [1, " + std::to_string(kMaxDim) + "], got " +
                          std::to_string(dim));
  }
}

slot_t VectorStore::add(const float* v) {
  if (count_ >= static_cast<std::size_t>(kInvalidSlot) - 1) throw InvalidArgument("vector store is full");
  const float* begin = data_.data();
  const float* end = data_.data() + data_.size();
  if (!data_.empty() && v >= begin && v < end) {
    // v aliases our own buffer, which push_back may reallocate: copy it first.
    const std::vector<float> tmp(v, v + dim_);
    data_.insert(data_.end(), tmp.begin(), tmp.end());
  } else {
    data_.insert(data_.end(), v, v + dim_);
  }
  return static_cast<slot_t>(count_++);
}

void VectorStore::set(slot_t s, const float* v) {
  float* dst = data_.data() + static_cast<std::size_t>(s) * dim_;
  if (dst != v) std::copy(v, v + dim_, dst);
}

void VectorStore::pop_back() {
  if (count_ == 0) return;
  --count_;
  data_.resize(count_ * dim_);
}

VectorStore VectorStore::from_raw(std::size_t dim, std::vector<float> data) {
  VectorStore s(dim);
  if (data.size() % dim != 0) throw FormatError("vector payload size is not a multiple of the dimension");
  s.count_ = data.size() / dim;
  s.data_ = std::move(data);
  return s;
}

}  // namespace hnsw
