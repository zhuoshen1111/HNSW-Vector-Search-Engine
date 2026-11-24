// CommandLine: the interactive / scripted front end of the vector search engine.
//
// It owns the active index through std::unique_ptr<Index> and dispatches text commands such as
//   build data.fvecs --metric l2 --M 16 --efc 200
//   query queries.fvecs --k 10 --ef 50 --gt gt.ivecs
//   delete 312
//   save index.bin
// Every command either succeeds or prints "error: ..." and leaves the index unchanged.
#pragma once

#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hnsw/index.h"

namespace hnsw {

class CommandLine {
 public:
  CommandLine(std::ostream& out, std::ostream& err);

  /// Runs one command line. Returns true on success (including blank/comment lines). On failure
  /// prints "error: <message>" to the error stream and returns false; the index is unchanged.
  bool execute(const std::string& line);

  /// Runs commands from `in` until EOF or `quit`. In interactive mode a prompt is printed and
  /// errors never stop the loop; otherwise each command is echoed as "> cmd" (if echo is on) and
  /// the first error stops the run unless keep_going is set. Returns the process exit code
  /// (0 = all commands succeeded, 1 = at least one command failed).
  int run(std::istream& in, bool interactive, bool keep_going);

  void set_timing(bool on) noexcept { timing_ = on; }
  void set_echo(bool on) noexcept { echo_ = on; }
  bool should_exit() const noexcept { return exit_; }
  const Index* index() const noexcept { return index_.get(); }

  /// Parsed "positional --option value --flag" arguments of one command.
  struct Args {
    std::string command;
    std::vector<std::string> positional;
    std::map<std::string, std::string> options;
  };

 private:
  using Handler = void (CommandLine::*)(const Args&);
  struct Spec {
    Handler handler;
    const char* usage;
    const char* summary;
  };
  static const std::map<std::string, Spec>& commands();

  void cmd_help(const Args& a);
  void cmd_build(const Args& a);
  void cmd_new(const Args& a);
  void cmd_insert(const Args& a);
  void cmd_add(const Args& a);
  void cmd_upsert(const Args& a);
  void cmd_delete(const Args& a);
  void cmd_query(const Args& a);
  void cmd_search(const Args& a);
  void cmd_save(const Args& a);
  void cmd_load(const Args& a);
  void cmd_info(const Args& a);
  void cmd_stats(const Args& a);
  void cmd_check(const Args& a);
  void cmd_compact(const Args& a);
  void cmd_set(const Args& a);
  void cmd_echo(const Args& a);
  void cmd_quit(const Args& a);

  Index& require_index() const;
  void insert_batch(Index& idx, const std::vector<label_t>& ids, const float* data, std::size_t n, std::size_t d);
  void print_results(const std::string& tag, const std::vector<SearchResult>& res, double ms);
  void note_label(label_t id) noexcept;

  std::ostream& out_;
  std::ostream& err_;
  std::unique_ptr<Index> index_;
  std::size_t default_k_ = 10;
  std::size_t default_ef_ = 50;
  bool timing_ = true;
  bool echo_ = true;
  bool exit_ = false;
  label_t next_id_ = 0;  ///< default first id for `insert` of an .fvecs file
};

/// Splits a command line into an Args structure (whitespace separated; double quotes group).
/// Throws InvalidArgument on unbalanced quotes.
CommandLine::Args parse_command_line(const std::string& line);

}  // namespace hnsw
