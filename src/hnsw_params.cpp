#include <cmath>
#include <sstream>

#include "hnsw/hnsw_index.h"

namespace hnsw {

const char* selection_name(NeighborSelection s) noexcept {
  return s == NeighborSelection::Simple ? "simple" : "heuristic";
}

NeighborSelection parse_selection(const std::string& s) {
  if (s == "simple") return NeighborSelection::Simple;
  if (s == "heuristic") return NeighborSelection::Heuristic;
  throw InvalidArgument("unknown neighbour selection '" + s + "' (expected simple or heuristic)");
}

const char* delete_policy_name(DeletePolicy p) noexcept {
  switch (p) {
    case DeletePolicy::Tombstone:
      return "tombstone";
    case DeletePolicy::RepairPrune:
      return "repair-prune";
    case DeletePolicy::Repair:
      return "repair";
    case DeletePolicy::RepairFixed:
      return "repair-fixed";
  }
  return "unknown";
}

DeletePolicy parse_delete_policy(const std::string& s) {
  if (s == "tombstone") return DeletePolicy::Tombstone;
  if (s == "repair") return DeletePolicy::Repair;
  if (s == "repair-prune") return DeletePolicy::RepairPrune;
  if (s == "repair-fixed") return DeletePolicy::RepairFixed;
  throw InvalidArgument("unknown delete policy '" + s + "' (expected tombstone, repair, repair-prune or repair-fixed)");
}

void HnswParams::validate() const {
  if (M < 2 || M > 4096) throw InvalidArgument("M must be in [2, 4096], got " + std::to_string(M));
  const std::size_t m0 = effective_M0();
  if (m0 < M || m0 > 8192) throw InvalidArgument("M0 must be in [M, 8192], got " + std::to_string(m0));
  if (ef_construction < 1 || ef_construction > (1u << 20)) {
    throw InvalidArgument("ef_construction must be in [1, 1048576], got " + std::to_string(ef_construction));
  }
}

int LevelGenerator::next() noexcept {
  // 53 random bits -> u in (0, 1]; independent of the standard library's distributions.
  const std::uint64_t bits = rng_() >> 11;
  const double u = (static_cast<double>(bits) + 1.0) * (1.0 / 9007199254740992.0);
  const double level = -std::log(u) * mult_;
  if (level >= static_cast<double>(kMaxLevel)) return kMaxLevel;
  return static_cast<int>(level);
}

std::string LevelGenerator::state() const {
  std::ostringstream os;
  os << rng_;
  return os.str();
}

void LevelGenerator::restore(const std::string& s) {
  std::istringstream is(s);
  std::mt19937_64 engine;
  is >> engine;
  if (is.fail()) throw FormatError("index file has a corrupt random-generator state");
  rng_ = engine;
}

}  // namespace hnsw
