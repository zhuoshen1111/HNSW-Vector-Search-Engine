// Vector file readers/writers.
//
//   .fvecs  TEXMEX format: per vector an int32 dimension d followed by d float32 values.
//           Ids are implicit row numbers (offset by a start id).
//   .ivecs  same layout with int32 values (ground-truth neighbour lists).
//   .csv    one vector per line: "id,v1,v2,...,vd". Blank lines and lines starting with '#'
//           are ignored. Every row must have the same dimension.
//
// All readers validate the input and throw IoError (cannot open) or FormatError (with the
// offending row / line number) for truncated files, inconsistent dimensions, bad numbers and
// NaN/inf components.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "hnsw/common.h"

namespace hnsw {

/// Dense row-major float matrix.
struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> data;

  const float* row(std::size_t i) const noexcept { return data.data() + i * cols; }
  float* row(std::size_t i) noexcept { return data.data() + i * cols; }
};

/// Vectors plus their external ids.
struct LabeledVectors {
  Matrix vectors;
  std::vector<label_t> ids;
};

inline constexpr std::size_t kNoLimit = std::numeric_limits<std::size_t>::max();

Matrix read_fvecs(const std::string& path, std::size_t limit = kNoLimit);
void write_fvecs(const std::string& path, const Matrix& m);

std::vector<std::vector<std::int32_t>> read_ivecs(const std::string& path, std::size_t limit = kNoLimit);
void write_ivecs(const std::string& path, const std::vector<std::vector<std::int32_t>>& rows);

LabeledVectors read_csv_vectors(const std::string& path, std::size_t limit = kNoLimit);
void write_csv_vectors(const std::string& path, const LabeledVectors& v);

/// Dispatches on the extension (.fvecs or .csv). For .fvecs, ids are start_id, start_id+1, ...
LabeledVectors read_vector_file(const std::string& path, std::size_t limit = kNoLimit, label_t start_id = 0);

/// Lower-cased file extension including the dot ("" if none).
std::string file_extension(const std::string& path);

}  // namespace hnsw
