// Save/load round trips, corruption handling and vector-file parsing.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "hnsw/index_factory.h"
#include "hnsw/io.h"
#include "hnsw/serializer.h"
#include "test_framework.h"
#include "test_util.h"

using namespace hnsw;
namespace fs = std::filesystem;

namespace {

std::string tmp_path(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / "hnsw_vs_tests";
  fs::create_directories(dir);
  return (dir / name).string();
}

std::vector<char> read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_all(const std::string& path, const std::vector<char>& bytes, std::size_t len) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(len));
}

std::unique_ptr<Index> build_index(IndexKind kind, Metric metric, std::size_t n, std::size_t d, std::uint64_t seed) {
  IndexConfig cfg;
  cfg.kind = kind;
  cfg.metric = metric;
  cfg.dim = d;
  cfg.hnsw = testutil::small_params();
  auto idx = make_index(cfg);
  const auto data = testutil::make_data(n, d, seed);
  for (std::size_t i = 0; i < n; ++i) idx->add(10 * i, data.data() + i * d, d);
  for (std::size_t i = 0; i < n; i += 7) idx->remove(10 * i);  // include tombstones in the file
  return idx;
}

}  // namespace

TEST(save_load_roundtrip_preserves_answers_all_kinds_and_metrics) {
  const std::size_t d = 12;
  const auto queries = testutil::make_data(40, d, 201);
  for (IndexKind kind : {IndexKind::BruteForce, IndexKind::HNSW}) {
    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
      auto idx = build_index(kind, m, 800, d, 202);
      const std::string path = tmp_path("roundtrip.idx");
      save_index(*idx, path);
      CHECK(!fs::exists(path + ".tmp"));  // temp file renamed away
      auto loaded = load_index(path);
      CHECK_EQ(loaded->kind(), kind);
      CHECK_EQ(loaded->metric(), m);
      CHECK_EQ(loaded->dim(), d);
      CHECK_EQ(loaded->size(), idx->size());
      CHECK(loaded->labels() == idx->labels());
      CHECK(loaded->check_invariants().empty());
      CHECK_EQ(loaded->stats().edges, idx->stats().edges);
      for (std::size_t q = 0; q < 40; ++q) {
        const std::vector<float> qv(queries.begin() + q * d, queries.begin() + (q + 1) * d);
        CHECK(loaded->knn(qv, 10, 40) == idx->knn(qv, 10, 40));
      }
    }
  }
}

TEST(save_load_then_continue_inserting_is_deterministic) {
  // The level RNG state is persisted, so insert-after-reload equals insert-without-reload.
  const std::size_t d = 8;
  auto a = build_index(IndexKind::HNSW, Metric::L2, 500, d, 211);
  auto b = build_index(IndexKind::HNSW, Metric::L2, 500, d, 211);
  const std::string path = tmp_path("continue.idx");
  save_index(*b, path);
  b = load_index(path);
  const auto more = testutil::make_data(300, d, 212);
  for (std::size_t i = 0; i < 300; ++i) {
    a->add(100000 + i, more.data() + i * d, d);
    b->add(100000 + i, more.data() + i * d, d);
  }
  const auto* ha = dynamic_cast<HNSWIndex<L2>*>(a.get());
  const auto* hb = dynamic_cast<HNSWIndex<L2>*>(b.get());
  REQUIRE(ha != nullptr && hb != nullptr);
  for (label_t id : ha->labels()) {
    CHECK_EQ(ha->level_of(id), hb->level_of(id));
    CHECK(ha->neighbors_of(id, 0) == hb->neighbors_of(id, 0));
  }
}

TEST(load_rejects_truncated_corrupt_and_foreign_files) {
  auto idx = build_index(IndexKind::HNSW, Metric::L2, 300, 6, 221);
  const std::string good = tmp_path("good.idx");
  save_index(*idx, good);
  const std::vector<char> bytes = read_all(good);
  REQUIRE(bytes.size() > 100);
  const std::string bad = tmp_path("bad.idx");
  // Every truncation point must be detected without crashing or over-allocating.
  for (std::size_t len : {std::size_t{0}, std::size_t{4}, std::size_t{8}, std::size_t{20}, std::size_t{33},
                          bytes.size() / 3, bytes.size() / 2, bytes.size() - 9, bytes.size() - 1}) {
    write_all(bad, bytes, len);
    CHECK_THROWS_AS(load_index(bad), FormatError);
  }
  // Flipped payload byte: caught by the checksum (or by structural validation).
  std::vector<char> flipped = bytes;
  flipped[bytes.size() / 2] = static_cast<char>(flipped[bytes.size() / 2] ^ 0x5A);
  write_all(bad, flipped, flipped.size());
  CHECK_THROWS_AS(load_index(bad), FormatError);
  // Bad magic.
  std::vector<char> magic = bytes;
  magic[0] = 'X';
  write_all(bad, magic, magic.size());
  CHECK_THROWS_AS(load_index(bad), FormatError);
  // Unsupported version (bytes 8..11).
  std::vector<char> version = bytes;
  version[8] = 9;
  write_all(bad, version, version.size());
  CHECK_THROWS_AS(load_index(bad), FormatError);
  // Trailing garbage.
  std::vector<char> longer = bytes;
  longer.push_back('!');
  write_all(bad, longer, longer.size());
  CHECK_THROWS_AS(load_index(bad), FormatError);
  // Missing file is an I/O error, not a format error.
  CHECK_THROWS_AS(load_index(tmp_path("does_not_exist.idx")), IoError);
  // The good file still loads.
  CHECK(load_index(good)->size() == idx->size());
}

TEST(failed_save_keeps_previous_file) {
  auto idx = build_index(IndexKind::BruteForce, Metric::L2, 50, 4, 231);
  const std::string path = tmp_path("keep.idx");
  save_index(*idx, path);
  const auto before = read_all(path);
  {
    BinaryWriter w(path);  // destroyed without commit, as if an exception escaped mid-save
    w.write_pod<std::uint64_t>(123);
  }
  CHECK(!fs::exists(path + ".tmp"));
  CHECK(read_all(path) == before);
  CHECK_THROWS_AS(save_index(*idx, tmp_path("no_such_dir/x.idx")), IoError);
}

TEST(fvecs_and_ivecs_roundtrip_and_errors) {
  Matrix m;
  m.rows = 3;
  m.cols = 4;
  m.data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::string f = tmp_path("m.fvecs");
  write_fvecs(f, m);
  const Matrix r = read_fvecs(f);
  CHECK_EQ(r.rows, 3u);
  CHECK_EQ(r.cols, 4u);
  CHECK(r.data == m.data);
  CHECK_EQ(read_fvecs(f, 2).rows, 2u);  // limit
  const auto bytes = read_all(f);
  write_all(tmp_path("trunc.fvecs"), bytes, bytes.size() - 2);
  CHECK_THROWS_AS(read_fvecs(tmp_path("trunc.fvecs")), FormatError);
  // Inconsistent dimension in the second row.
  std::vector<char> mixed = bytes;
  const std::int32_t three = 3;
  std::memcpy(mixed.data() + 20, &three, 4);
  write_all(tmp_path("mixed.fvecs"), mixed, mixed.size());
  CHECK_THROWS_AS(read_fvecs(tmp_path("mixed.fvecs")), FormatError);
  // NaN component.
  Matrix nan = m;
  nan.data[5] = std::nanf("");
  write_fvecs(tmp_path("nan.fvecs"), nan);
  CHECK_THROWS_AS(read_fvecs(tmp_path("nan.fvecs")), FormatError);
  CHECK_THROWS_AS(read_fvecs(tmp_path("missing.fvecs")), IoError);

  const std::vector<std::vector<std::int32_t>> gt = {{1, 2, 3}, {4, 5, 6}};
  write_ivecs(tmp_path("gt.ivecs"), gt);
  CHECK(read_ivecs(tmp_path("gt.ivecs")) == gt);
}

TEST(csv_parsing_valid_comments_and_errors) {
  const std::string ok = tmp_path("ok.csv");
  {
    std::ofstream out(ok);
    out << "# comment line\n\n7, 1.5, -2\n3,0,1e-3\n";
  }
  const LabeledVectors v = read_csv_vectors(ok);
  CHECK_EQ(v.vectors.rows, 2u);
  CHECK_EQ(v.vectors.cols, 2u);
  CHECK(v.ids == std::vector<label_t>({7, 3}));
  CHECK_EQ(v.vectors.row(0)[1], -2.0f);
  const LabeledVectors via_ext = read_vector_file(ok);
  CHECK(via_ext.ids == v.ids);

  auto expect_bad = [](const std::string& name, const std::string& text) {
    const std::string p = tmp_path(name);
    {
      std::ofstream out(p);
      out << text;
    }
    CHECK_THROWS_AS(read_csv_vectors(p), FormatError);
  };
  expect_bad("bad1.csv", "1,2,3\n2,4\n");        // inconsistent dimension
  expect_bad("bad2.csv", "x,1,2\n");             // non-numeric id
  expect_bad("bad3.csv", "-1,1,2\n");            // negative id
  expect_bad("bad4.csv", "1,abc,2\n");           // non-numeric component
  expect_bad("bad5.csv", "1\n");                 // no components
  expect_bad("bad6.csv", "1,nan,2\n");           // NaN
  expect_bad("bad7.csv", "1,2,\n");              // trailing comma
  CHECK_THROWS_AS(read_vector_file(tmp_path("x.txt2")), InvalidArgument);
  CHECK_EQ(file_extension("a/b.c/data.FVECS"), std::string(".fvecs"));
  CHECK_EQ(file_extension("a.d/noext"), std::string(""));
}
