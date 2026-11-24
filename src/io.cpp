#include "hnsw/io.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace hnsw {

namespace {

std::ifstream open_in(const std::string& path, bool binary) {
  std::ifstream in(path, binary ? std::ios::binary : std::ios::in);
  if (!in) throw IoError("cannot open '" + path + "'");
  return in;
}

std::ofstream open_out(const std::string& path, bool binary) {
  std::ofstream out(path, binary ? (std::ios::binary | std::ios::trunc) : std::ios::trunc);
  if (!out) throw IoError("cannot open '" + path + "' for writing");
  return out;
}

std::uint64_t file_size(std::ifstream& in) {
  in.seekg(0, std::ios::end);
  const std::streamoff end = in.tellg();
  in.seekg(0, std::ios::beg);
  return end < 0 ? 0 : static_cast<std::uint64_t>(end);
}

/// Shared reader for .fvecs / .ivecs: calls on_row(row_index, dim) after reading each header and
/// expects the callback to consume exactly dim * 4 bytes.
template <typename OnRow>
void read_vecs(const std::string& path, std::size_t limit, OnRow on_row) {
  std::ifstream in = open_in(path, true);
  const std::uint64_t size = file_size(in);
  std::uint64_t pos = 0;
  std::int32_t expected = -1;
  for (std::size_t row = 0; row < limit && pos < size; ++row) {
    std::int32_t d = 0;
    if (size - pos < sizeof(d)) throw FormatError("'" + path + "' is truncated in the header of vector " + std::to_string(row));
    in.read(reinterpret_cast<char*>(&d), sizeof(d));
    pos += sizeof(d);
    if (d <= 0 || static_cast<std::size_t>(d) > kMaxDim) {
      throw FormatError("'" + path + "': vector " + std::to_string(row) + " has invalid dimension " + std::to_string(d));
    }
    if (expected < 0) expected = d;
    if (d != expected) {
      throw FormatError("'" + path + "': vector " + std::to_string(row) + " has dimension " + std::to_string(d) +
                        ", expected " + std::to_string(expected));
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(d) * 4;
    if (size - pos < bytes) throw FormatError("'" + path + "' is truncated inside vector " + std::to_string(row));
    on_row(in, row, static_cast<std::size_t>(d));
    if (!in) throw FormatError("read error in '" + path + "'");
    pos += bytes;
  }
}

void check_finite_row(const std::string& path, const float* v, std::size_t d, std::size_t row) {
  for (std::size_t j = 0; j < d; ++j) {
    if (!std::isfinite(v[j])) {
      throw FormatError("'" + path + "': vector " + std::to_string(row) + " component " + std::to_string(j) +
                        " is NaN or infinite");
    }
  }
}

std::string trim(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

}  // namespace

std::string file_extension(const std::string& path) {
  const std::size_t slash = path.find_last_of("/\\");
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
  std::string ext = path.substr(dot);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

Matrix read_fvecs(const std::string& path, std::size_t limit) {
  Matrix m;
  read_vecs(path, limit, [&](std::ifstream& in, std::size_t row, std::size_t d) {
    if (m.cols == 0) m.cols = d;
    m.data.resize((row + 1) * d);
    in.read(reinterpret_cast<char*>(m.row(row)), static_cast<std::streamsize>(d * sizeof(float)));
    check_finite_row(path, m.row(row), d, row);
    m.rows = row + 1;
  });
  return m;
}

void write_fvecs(const std::string& path, const Matrix& m) {
  std::ofstream out = open_out(path, true);
  const auto d = static_cast<std::int32_t>(m.cols);
  for (std::size_t i = 0; i < m.rows; ++i) {
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(m.row(i)), static_cast<std::streamsize>(m.cols * sizeof(float)));
  }
  if (!out) throw IoError("write to '" + path + "' failed");
}

std::vector<std::vector<std::int32_t>> read_ivecs(const std::string& path, std::size_t limit) {
  std::vector<std::vector<std::int32_t>> rows;
  read_vecs(path, limit, [&](std::ifstream& in, std::size_t /*row*/, std::size_t d) {
    rows.emplace_back(d);
    in.read(reinterpret_cast<char*>(rows.back().data()), static_cast<std::streamsize>(d * sizeof(std::int32_t)));
  });
  return rows;
}

void write_ivecs(const std::string& path, const std::vector<std::vector<std::int32_t>>& rows) {
  std::ofstream out = open_out(path, true);
  for (const auto& r : rows) {
    const auto d = static_cast<std::int32_t>(r.size());
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(r.data()), static_cast<std::streamsize>(r.size() * sizeof(std::int32_t)));
  }
  if (!out) throw IoError("write to '" + path + "' failed");
}

LabeledVectors read_csv_vectors(const std::string& path, std::size_t limit) {
  std::ifstream in = open_in(path, false);
  LabeledVectors out;
  std::string line;
  std::size_t line_no = 0;
  while (out.ids.size() < limit && std::getline(in, line)) {
    ++line_no;
    const std::string t = trim(line);
    if (t.empty() || t[0] == '#') continue;
    auto where = [&]() { return "'" + path + "' line " + std::to_string(line_no) + ": "; };
    std::vector<std::string> fields;
    std::stringstream ss(t);
    std::string f;
    while (std::getline(ss, f, ',')) fields.push_back(trim(f));
    if (!t.empty() && t.back() == ',') fields.push_back("");
    if (fields.size() < 2) throw FormatError(where() + "expected 'id,v1,...,vd'");
    const std::string& id_s = fields[0];
    if (id_s.empty() || !std::all_of(id_s.begin(), id_s.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
      throw FormatError(where() + "id '" + id_s + "' is not a non-negative integer");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long id = std::strtoull(id_s.c_str(), &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0') throw FormatError(where() + "id '" + id_s + "' is out of range");
    const std::size_t d = fields.size() - 1;
    if (d > kMaxDim) throw FormatError(where() + "too many components");
    if (out.vectors.cols == 0) out.vectors.cols = d;
    if (d != out.vectors.cols) {
      throw FormatError(where() + "row has " + std::to_string(d) + " components, expected " +
                        std::to_string(out.vectors.cols));
    }
    for (std::size_t j = 1; j <= d; ++j) {
      const std::string& s = fields[j];
      errno = 0;
      end = nullptr;
      const float v = s.empty() ? 0.0f : std::strtof(s.c_str(), &end);
      if (s.empty() || end == nullptr || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        throw FormatError(where() + "component " + std::to_string(j - 1) + " ('" + s + "') is not a finite number");
      }
      out.vectors.data.push_back(v);
    }
    out.ids.push_back(static_cast<label_t>(id));
    ++out.vectors.rows;
  }
  return out;
}

void write_csv_vectors(const std::string& path, const LabeledVectors& v) {
  std::ofstream out = open_out(path, false);
  out.precision(9);
  for (std::size_t i = 0; i < v.vectors.rows; ++i) {
    out << v.ids[i];
    const float* r = v.vectors.row(i);
    for (std::size_t j = 0; j < v.vectors.cols; ++j) out << ',' << r[j];
    out << '\n';
  }
  if (!out) throw IoError("write to '" + path + "' failed");
}

LabeledVectors read_vector_file(const std::string& path, std::size_t limit, label_t start_id) {
  const std::string ext = file_extension(path);
  if (ext == ".csv" || ext == ".txt") return read_csv_vectors(path, limit);
  if (ext == ".fvecs") {
    LabeledVectors out;
    out.vectors = read_fvecs(path, limit);
    out.ids.resize(out.vectors.rows);
    for (std::size_t i = 0; i < out.ids.size(); ++i) out.ids[i] = start_id + i;
    return out;
  }
  throw InvalidArgument("unsupported vector file '" + path + "' (expected .fvecs or .csv)");
}

}  // namespace hnsw
