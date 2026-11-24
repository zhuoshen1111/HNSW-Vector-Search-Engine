// Runtime construction, persistence and loading of indexes behind the Index interface.
//
// The metric is a compile-time template parameter of each index class; these functions map
// the runtime choice (CLI flag or file header) onto the right instantiation.
//
// Index file format (version 1, little-endian):
//   magic "EE538HNS" | u32 version | u32 endian marker 0x01020304 | u8 kind | u8 metric | u64 dim
//   kind-specific body (see BruteForceIndex::save_body / HNSWIndex::save_body)
//   u64 FNV-1a checksum of every preceding byte
#pragma once

#include <memory>
#include <string>

#include "hnsw/hnsw_index.h"
#include "hnsw/index.h"

namespace hnsw {

struct IndexConfig {
  IndexKind kind = IndexKind::HNSW;
  Metric metric = Metric::L2;
  std::size_t dim = 0;
  HnswParams hnsw;  ///< used when kind == HNSW (single_layer=true gives the NSW baseline)
};

inline constexpr std::uint32_t kIndexFormatVersion = 1;

/// Throws InvalidArgument for an invalid configuration.
std::unique_ptr<Index> make_index(const IndexConfig& config);

/// Atomically writes the index to path (temp file + rename). Throws IoError on failure.
void save_index(const Index& index, const std::string& path);

/// Reads and fully validates an index file. Throws IoError (cannot open) or FormatError
/// (bad magic, unsupported version, truncated, checksum mismatch, inconsistent structure).
std::unique_ptr<Index> load_index(const std::string& path);

}  // namespace hnsw
