#include "hnsw/index_factory.h"

#include <cstring>

#include "hnsw/brute_force_index.h"
#include "hnsw/hnsw_index.h"

namespace hnsw {

// Explicit instantiations: the templates are compiled once here; headers declare them extern.
template class BruteForceIndex<L2>;
template class BruteForceIndex<InnerProduct>;
template class BruteForceIndex<Cosine>;
template class HNSWIndex<L2>;
template class HNSWIndex<InnerProduct>;
template class HNSWIndex<Cosine>;

namespace {

constexpr char kMagic[8] = {'E', 'E', '5', '3', '8', 'H', 'N', 'S'};
constexpr std::uint32_t kEndianMarker = 0x01020304u;

template <template <typename> class IndexT, typename... Args>
std::unique_ptr<Index> construct(Metric m, Args&&... args) {
  switch (m) {
    case Metric::L2:
      return std::make_unique<IndexT<L2>>(std::forward<Args>(args)...);
    case Metric::InnerProduct:
      return std::make_unique<IndexT<InnerProduct>>(std::forward<Args>(args)...);
    case Metric::Cosine:
      return std::make_unique<IndexT<Cosine>>(std::forward<Args>(args)...);
  }
  throw InvalidArgument("unknown metric");
}

template <template <typename> class IndexT>
std::unique_ptr<Index> load_kind(Metric m, BinaryReader& r, std::size_t dim) {
  switch (m) {
    case Metric::L2:
      return IndexT<L2>::load_body(r, dim);
    case Metric::InnerProduct:
      return IndexT<InnerProduct>::load_body(r, dim);
    case Metric::Cosine:
      return IndexT<Cosine>::load_body(r, dim);
  }
  throw FormatError("unknown metric");
}

}  // namespace

std::unique_ptr<Index> make_index(const IndexConfig& config) {
  if (config.dim == 0 || config.dim > kMaxDim) {
    throw InvalidArgument("dimension must be in [1, " + std::to_string(kMaxDim) + "]");
  }
  if (config.kind == IndexKind::BruteForce) return construct<BruteForceIndex>(config.metric, config.dim);
  config.hnsw.validate();
  return construct<HNSWIndex>(config.metric, config.dim, config.hnsw);
}

void save_index(const Index& index, const std::string& path) {
  BinaryWriter w(path);
  w.write_bytes(kMagic, sizeof(kMagic));
  w.write_pod<std::uint32_t>(kIndexFormatVersion);
  w.write_pod<std::uint32_t>(kEndianMarker);
  w.write_pod<std::uint8_t>(static_cast<std::uint8_t>(index.kind()));
  w.write_pod<std::uint8_t>(static_cast<std::uint8_t>(index.metric()));
  w.write_pod<std::uint64_t>(index.dim());
  index.save_body(w);
  w.commit();
}

std::unique_ptr<Index> load_index(const std::string& path) {
  BinaryReader r(path);
  char magic[sizeof(kMagic)];
  if (r.payload_remaining() < sizeof(kMagic)) throw FormatError("'" + path + "' is not an index file (too small)");
  r.read_bytes(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw FormatError("'" + path + "' is not an index file (bad magic number)");
  }
  const auto version = r.read_pod<std::uint32_t>();
  if (version != kIndexFormatVersion) {
    throw FormatError("unsupported index format version " + std::to_string(version) + " (this build reads version " +
                      std::to_string(kIndexFormatVersion) + ")");
  }
  if (r.read_pod<std::uint32_t>() != kEndianMarker) throw FormatError("index file was written with another byte order");
  const auto kind = r.read_pod<std::uint8_t>();
  const auto metric = r.read_pod<std::uint8_t>();
  if (kind > 1) throw FormatError("index file has an unknown index kind " + std::to_string(kind));
  if (metric > 2) throw FormatError("index file has an unknown metric " + std::to_string(metric));
  const auto dim = static_cast<std::size_t>(r.read_u64_in_range("dimension", 1, kMaxDim));
  std::unique_ptr<Index> idx = static_cast<IndexKind>(kind) == IndexKind::BruteForce
                                   ? load_kind<BruteForceIndex>(static_cast<Metric>(metric), r, dim)
                                   : load_kind<HNSWIndex>(static_cast<Metric>(metric), r, dim);
  r.finish();
  return idx;
}

}  // namespace hnsw
