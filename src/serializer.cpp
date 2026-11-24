#include "hnsw/serializer.h"

#include <cstdio>

namespace hnsw {

std::uint64_t fnv1a_update(std::uint64_t hash, const void* data, std::size_t n) noexcept {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    hash ^= p[i];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

// ------------------------------------------------------------------------------------------
BinaryWriter::BinaryWriter(const std::string& path) : path_(path), tmp_path_(path + ".tmp") {
  out_.open(tmp_path_, std::ios::binary | std::ios::trunc);
  if (!out_) throw IoError("cannot open '" + tmp_path_ + "' for writing");
}

BinaryWriter::~BinaryWriter() {
  if (!committed_) {
    out_.close();
    std::remove(tmp_path_.c_str());
  }
}

void BinaryWriter::write_bytes(const void* data, std::size_t n) {
  if (n == 0) return;
  out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
  if (!out_) throw IoError("write to '" + tmp_path_ + "' failed (disk full?)");
  checksum_ = fnv1a_update(checksum_, data, n);
  written_ += n;
}

void BinaryWriter::write_string(const std::string& s) {
  write_pod<std::uint64_t>(s.size());
  write_bytes(s.data(), s.size());
}

void BinaryWriter::commit() {
  if (committed_) return;
  const std::uint64_t sum = checksum_;
  out_.write(reinterpret_cast<const char*>(&sum), sizeof(sum));
  out_.flush();
  if (!out_) throw IoError("write to '" + tmp_path_ + "' failed (disk full?)");
  out_.close();
  if (out_.fail()) throw IoError("closing '" + tmp_path_ + "' failed");
  if (std::rename(tmp_path_.c_str(), path_.c_str()) != 0) {
    throw IoError("cannot rename '" + tmp_path_ + "' to '" + path_ + "'");
  }
  committed_ = true;
}

// ------------------------------------------------------------------------------------------
BinaryReader::BinaryReader(const std::string& path) : path_(path) {
  in_.open(path, std::ios::binary);
  if (!in_) throw IoError("cannot open index file '" + path + "'");
  in_.seekg(0, std::ios::end);
  const std::streamoff end = in_.tellg();
  in_.seekg(0, std::ios::beg);
  if (end < 0) throw IoError("cannot determine the size of '" + path + "'");
  size_ = static_cast<std::uint64_t>(end);
  if (size_ < 8) throw FormatError("'" + path + "' is too small to be an index file");
}

void BinaryReader::read_bytes(void* dst, std::size_t n) {
  if (n == 0) return;
  if (n > payload_remaining()) {
    throw FormatError("index file '" + path_ + "' is truncated (needed " + std::to_string(n) + " more bytes)");
  }
  in_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (static_cast<std::size_t>(in_.gcount()) != n) throw FormatError("read from '" + path_ + "' failed");
  checksum_ = fnv1a_update(checksum_, dst, n);
  pos_ += n;
}

std::string BinaryReader::read_string(std::size_t max_len) {
  const auto len = read_pod<std::uint64_t>();
  if (len > max_len || len > payload_remaining()) throw FormatError("index file has a string field of invalid length");
  std::string s(static_cast<std::size_t>(len), '\0');
  if (len > 0) read_bytes(&s[0], static_cast<std::size_t>(len));
  return s;
}

std::uint64_t BinaryReader::read_u64_in_range(const char* field, std::uint64_t lo, std::uint64_t hi) {
  const auto v = read_pod<std::uint64_t>();
  if (v < lo || v > hi) {
    throw FormatError(std::string("index file field '") + field + "' = " + std::to_string(v) + " is out of range");
  }
  return v;
}

void BinaryReader::finish() {
  if (payload_remaining() != 0) {
    throw FormatError("index file '" + path_ + "' has " + std::to_string(payload_remaining()) +
                      " unexpected trailing bytes");
  }
  std::uint64_t stored = 0;
  in_.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (in_.gcount() != static_cast<std::streamsize>(sizeof(stored))) throw FormatError("cannot read checksum");
  if (stored != checksum_) throw FormatError("index file '" + path_ + "' is corrupt (checksum mismatch)");
}

}  // namespace hnsw
