// Binary persistence primitives.
//
// BinaryWriter writes to "<path>.tmp", accumulates an FNV-1a-64 checksum of every byte, and on
// commit() appends the checksum and atomically renames the temp file over <path>. If the writer
// is destroyed without commit() (for example because an exception escaped), the temp file is
// removed and any previous file at <path> is left untouched (RAII).
//
// BinaryReader bounds-checks every read against the file size before allocating, so truncated
// or corrupt headers produce FormatError instead of huge allocations or out-of-bounds reads.
// finish() verifies the trailing checksum and that no unexpected bytes remain.
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "hnsw/common.h"

namespace hnsw {

std::uint64_t fnv1a_update(std::uint64_t hash, const void* data, std::size_t n) noexcept;
inline constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;

class BinaryWriter {
 public:
  explicit BinaryWriter(const std::string& path);
  ~BinaryWriter();
  BinaryWriter(const BinaryWriter&) = delete;
  BinaryWriter& operator=(const BinaryWriter&) = delete;

  void write_bytes(const void* data, std::size_t n);

  template <typename T>
  void write_pod(const T& value) {
    static_assert(std::is_trivially_copyable<T>::value, "write_pod needs a trivially copyable type");
    write_bytes(&value, sizeof(T));
  }

  /// Writes the elements only; the element count must be recoverable from earlier fields.
  template <typename T>
  void write_array(const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable<T>::value, "write_array needs a trivially copyable type");
    if (!v.empty()) write_bytes(v.data(), v.size() * sizeof(T));
  }

  /// u64 length followed by the characters.
  void write_string(const std::string& s);

  /// Appends the checksum, flushes, closes and renames the temp file over the target path.
  void commit();

  std::uint64_t bytes_written() const noexcept { return written_; }

 private:
  std::string path_;
  std::string tmp_path_;
  std::ofstream out_;
  std::uint64_t checksum_ = kFnvOffset;
  std::uint64_t written_ = 0;
  bool committed_ = false;
};

class BinaryReader {
 public:
  explicit BinaryReader(const std::string& path);
  BinaryReader(const BinaryReader&) = delete;
  BinaryReader& operator=(const BinaryReader&) = delete;

  void read_bytes(void* dst, std::size_t n);

  template <typename T>
  T read_pod() {
    static_assert(std::is_trivially_copyable<T>::value, "read_pod needs a trivially copyable type");
    T value;
    read_bytes(&value, sizeof(T));
    return value;
  }

  /// Reads `count` elements after checking that enough bytes remain.
  template <typename T>
  std::vector<T> read_array(std::uint64_t count) {
    static_assert(std::is_trivially_copyable<T>::value, "read_array needs a trivially copyable type");
    if (count > payload_remaining() / sizeof(T)) {
      throw FormatError("index file '" + path_ + "' is truncated or corrupt (array of " +
                        std::to_string(count) + " elements exceeds remaining bytes)");
    }
    std::vector<T> out(static_cast<std::size_t>(count));
    if (count > 0) read_bytes(out.data(), static_cast<std::size_t>(count) * sizeof(T));
    return out;
  }

  std::string read_string(std::size_t max_len);

  /// Reads a u64 and checks lo <= value <= hi.
  std::uint64_t read_u64_in_range(const char* field, std::uint64_t lo, std::uint64_t hi);

  /// Bytes left before the 8-byte checksum footer.
  std::uint64_t payload_remaining() const noexcept { return size_ - 8 - pos_; }

  /// Verifies the checksum footer and that the payload was consumed exactly.
  void finish();

  const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
  std::ifstream in_;
  std::uint64_t size_ = 0;
  std::uint64_t pos_ = 0;
  std::uint64_t checksum_ = kFnvOffset;
};

}  // namespace hnsw
