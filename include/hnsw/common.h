// Common types, result ordering and the exception hierarchy shared by every module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hnsw {

/// External, user-visible vector identifier.
using label_t = std::uint64_t;
/// Internal dense slot number: an index into the contiguous per-vector arrays.
using slot_t = std::uint32_t;

inline constexpr slot_t kInvalidSlot = 0xFFFFFFFFu;
/// Largest accepted vector dimension (guards file parsing against absurd headers).
inline constexpr std::size_t kMaxDim = 1u << 16;

/// One ranked answer: external id plus distance under the index metric.
struct SearchResult {
  label_t id = 0;
  float distance = 0.0f;
};

inline bool operator==(const SearchResult& a, const SearchResult& b) noexcept {
  return a.id == b.id && a.distance == b.distance;
}
inline bool operator!=(const SearchResult& a, const SearchResult& b) noexcept { return !(a == b); }

/// The single total order used by every index: smaller distance first, ties broken by smaller id.
struct ResultOrder {
  bool operator()(const SearchResult& a, const SearchResult& b) const noexcept {
    if (a.distance < b.distance) return true;
    if (b.distance < a.distance) return false;
    return a.id < b.id;
  }
};

/// Base class of all errors reported by the library. Every operation that throws one of
/// these leaves the index unchanged (strong guarantee) unless documented otherwise.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
/// Bad caller input: dimension mismatch, duplicate id, zero vector under cosine, bad option.
class InvalidArgument : public Error {
 public:
  using Error::Error;
};
/// Unknown id (for example, deleting an id that is not live).
class NotFound : public Error {
 public:
  using Error::Error;
};
/// Malformed, truncated, corrupt or unsupported file contents.
class FormatError : public Error {
 public:
  using Error::Error;
};
/// The operating system refused an open/read/write/rename.
class IoError : public Error {
 public:
  using Error::Error;
};

}  // namespace hnsw
