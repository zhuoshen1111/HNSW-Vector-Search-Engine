// CommandLine front end: argument parsing, command workflows and the "error leaves the index
// unchanged" contract. Runs in-process with string streams; expects to be run from the repo root.
#include <filesystem>
#include <fstream>
#include <sstream>

#include "hnsw/cli.h"
#include "hnsw/io.h"
#include "test_framework.h"

using namespace hnsw;
namespace fs = std::filesystem;

namespace {

struct Session {
  std::ostringstream out;
  std::ostringstream err;
  CommandLine cli{out, err};
  Session() { cli.set_timing(false); }
  bool run(const std::string& cmd) {
    out.str("");
    err.str("");
    return cli.execute(cmd);
  }
};

std::string tmp_path(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / "hnsw_vs_tests";
  fs::create_directories(dir);
  return (dir / name).string();
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

}  // namespace

TEST(cli_parse_command_line) {
  const auto a = parse_command_line("Query  data.fvecs --k 5 --exact --ef 20 \"my file.csv\"");
  CHECK_EQ(a.command, std::string("query"));
  CHECK(a.positional == std::vector<std::string>({"data.fvecs", "my file.csv"}));
  CHECK_EQ(a.options.at("k"), std::string("5"));
  CHECK_EQ(a.options.at("exact"), std::string("1"));
  CHECK_EQ(a.options.at("ef"), std::string("20"));
  CHECK_THROWS_AS(parse_command_line("build \"unterminated"), InvalidArgument);
  CHECK_THROWS_AS(parse_command_line("query x --k 1 --k 2"), InvalidArgument);
  CHECK(parse_command_line("   ").command.empty());
}

TEST(cli_build_query_delete_save_load_workflow) {
  Session s;
  CHECK(s.run("# comments and blank lines are ignored"));
  CHECK(s.run(""));
  CHECK(!s.run("query tests/data/tiny_queries.csv"));  // no index yet
  CHECK(s.err.str().find("no index loaded") != std::string::npos);
  REQUIRE(s.run("build tests/data/tiny.csv --M 4 --efc 16"));
  CHECK_EQ(s.cli.index()->size(), 8u);
  REQUIRE(s.run("query tests/data/tiny_queries.csv --k 3 --exact"));
  CHECK(s.out.str().find("q0: (3, 0.05) (1, 0.65) (2, 0.85)") != std::string::npos);
  CHECK(s.out.str().find("recall@3 1.0000") != std::string::npos);
  REQUIRE(s.run("delete 3"));
  REQUIRE(s.run("search 0.9,0.8 --k 2"));
  CHECK(s.out.str().find("result: (1, 0.65) (2, 0.85)") != std::string::npos);
  const std::string path = tmp_path("cli.idx");
  REQUIRE(s.run("save " + path));
  REQUIRE(s.run("add 100 0.9,0.8"));
  CHECK_EQ(s.cli.index()->size(), 8u);
  REQUIRE(s.run("load " + path));
  CHECK_EQ(s.cli.index()->size(), 7u);
  CHECK(!s.cli.index()->contains(100));
  REQUIRE(s.run("upsert 1 9,9"));
  CHECK(s.out.str().find("replaced id 1") != std::string::npos);
  REQUIRE(s.run("check"));
  REQUIRE(s.run("compact"));
  CHECK(s.out.str().find("removed 2 tombstones") != std::string::npos);
  REQUIRE(s.run("info"));
  CHECK(s.out.str().find("7 live vectors, 0 tombstones") != std::string::npos);
  REQUIRE(s.run("stats"));
  CHECK(s.out.str().find("bytes/vector") != std::string::npos);
  REQUIRE(s.run("help"));
  REQUIRE(s.run("help query"));
  REQUIRE(s.run("set ef 30"));
  REQUIRE(s.run("quit"));
  CHECK(s.cli.should_exit());
}

TEST(cli_errors_leave_the_index_unchanged) {
  Session s;
  const std::string base = tmp_path("cos_base.csv");
  write_text(base, "1,1,0\n2,0,1\n3,1,1\n4,2,1\n5,1,2\n6,3,3\n");
  REQUIRE(s.run("build " + base + " --M 4 --efc 16 --metric cosine"));
  const auto labels = s.cli.index()->labels();
  // A batch with a bad row must not insert any of its rows.
  const std::string batch = tmp_path("batch.csv");
  write_text(batch, "100,1,2\n101,0,0\n102,3,4\n");  // row 1 is a zero vector under cosine
  CHECK(!s.run("insert " + batch));
  CHECK(s.err.str().find("row 1") != std::string::npos);
  write_text(batch, "100,1,2\n5,3,4\n");  // id 5 already exists
  CHECK(!s.run("insert " + batch));
  write_text(batch, "100,1,2\n100,3,4\n");  // duplicate within the file
  CHECK(!s.run("insert " + batch));
  write_text(batch, "100,1,2,3\n");  // dimension mismatch
  CHECK(!s.run("insert " + batch));
  CHECK(!s.run("delete 1 999"));  // 999 unknown: id 1 must not be deleted either
  CHECK(!s.run("delete 2 2"));
  CHECK(!s.run("add 50 1,2,3"));
  CHECK(!s.run("add 51 0,0"));
  CHECK(!s.run("add x 1,2"));
  CHECK(!s.run("upsert 4 0,0"));  // invalid replacement must keep the old vector
  CHECK(s.cli.index()->contains(4));
  CHECK(!s.run("query tests/data/tiny_queries.csv --k -3"));
  CHECK(!s.run("query tests/data/tiny_queries.csv --bogus 1"));
  CHECK(!s.run("query tests/data/tiny_queries.csv --k"));
  CHECK(!s.run("frobnicate"));
  CHECK(!s.run("load " + tmp_path("missing.idx")));
  const std::string garbage = tmp_path("garbage.idx");
  write_text(garbage, "this is not an index file at all");
  CHECK(!s.run("load " + garbage));
  CHECK(!s.run("build tests/data/does_not_exist.csv"));
  CHECK(!s.run("build tests/data/tiny.csv --M 1"));
  CHECK(!s.run("build tests/data/tiny.csv --metric hamming"));
  CHECK(s.cli.index()->labels() == labels);  // nothing above changed the index
  CHECK(s.cli.index()->metric() == Metric::Cosine);
  REQUIRE(s.run("check"));
}

TEST(cli_query_with_ground_truth_file_and_output) {
  Session s;
  REQUIRE(s.run("build tests/data/tiny.csv --index bruteforce"));
  const std::string gt = tmp_path("tiny_gt.ivecs");
  write_ivecs(gt, {{3, 1, 2}, {4, 5, 6}, {7, 5, 6}});
  const std::string out = tmp_path("tiny_results.csv");
  REQUIRE(s.run("query tests/data/tiny_queries.csv --k 3 --gt " + gt + " --out " + out));
  CHECK(s.out.str().find("recall@3 1.0000") != std::string::npos);
  std::ifstream in(out);
  std::string header, first;
  std::getline(in, header);
  std::getline(in, first);
  CHECK_EQ(header, std::string("query,rank,id,distance"));
  CHECK_EQ(first, std::string("0,0,3,0.0500000007"));
  CHECK(!s.run("query tests/data/tiny_queries.csv --k 5 --gt " + gt));  // gt rows too short
  CHECK(!s.run("query tests/data/tiny_queries.csv --gt " + gt + " --exact"));
}

TEST(cli_run_script_modes_and_exit_codes) {
  std::ostringstream out, err;
  CommandLine cli(out, err);
  cli.set_timing(false);
  std::istringstream ok("build tests/data/tiny.csv --M 4\ninfo\n");
  CHECK_EQ(cli.run(ok, false, false), 0);
  CHECK(out.str().find("> build tests/data/tiny.csv --M 4") != std::string::npos);  // echoed
  std::istringstream stop("info\nbogus\ninfo\n");
  out.str("");
  CHECK_EQ(cli.run(stop, false, false), 1);
  const std::string transcript = out.str();
  CHECK_EQ(static_cast<int>(std::count(transcript.begin(), transcript.end(), '\n')), 3);  // stopped after bogus
  std::istringstream go("bogus\ninfo\n");
  out.str("");
  CHECK_EQ(cli.run(go, false, true), 1);  // keep going, still report failure
  CHECK(out.str().find("hnsw(") != std::string::npos);
}
