#include "hnsw/cli.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#include "hnsw/brute_force_index.h"
#include "hnsw/hash_table.h"
#include "hnsw/index_factory.h"
#include "hnsw/io.h"
#include "hnsw/timer.h"

namespace hnsw {

namespace {

const std::set<std::string>& boolean_flags() {
  static const std::set<std::string> flags = {"exact"};
  return flags;
}

std::string trim(const std::string& s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::uint64_t parse_u64(const std::string& s, const std::string& what) {
  const bool digits = !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
  if (!digits) throw InvalidArgument(what + " must be a non-negative integer, got '" + s + "'");
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
  if (errno == ERANGE || end == nullptr || *end != '\0') throw InvalidArgument(what + " '" + s + "' is out of range");
  return static_cast<std::uint64_t>(v);
}

std::vector<float> parse_float_list(const std::string& s) {
  std::vector<float> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trim(tok);
    errno = 0;
    char* end = nullptr;
    const float v = tok.empty() ? 0.0f : std::strtof(tok.c_str(), &end);
    if (tok.empty() || end == nullptr || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
      throw InvalidArgument("'" + tok + "' is not a finite number (vectors are written as v1,v2,...,vd)");
    }
    out.push_back(v);
  }
  if (out.empty() || (!s.empty() && s.back() == ',')) throw InvalidArgument("malformed vector '" + s + "'");
  return out;
}

std::string fmt_dist(float d) {
  std::ostringstream os;
  os << std::setprecision(6) << d;
  return os.str();
}

std::string fmt_fixed(double v, int digits) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(digits) << v;
  return os.str();
}

/// Validates option names of one command and converts option values.
class Options {
 public:
  Options(const CommandLine::Args& a, std::initializer_list<const char*> allowed) : args_(a) {
    std::set<std::string> ok;
    for (const char* name : allowed) ok.insert(name);
    for (const auto& kv : a.options) {
      if (ok.count(kv.first) == 0) throw InvalidArgument("unknown option --" + kv.first + " for '" + a.command + "'");
      if (boolean_flags().count(kv.first) == 0 && kv.second.empty()) {
        throw InvalidArgument("option --" + kv.first + " needs a value");
      }
    }
  }
  bool has(const std::string& name) const { return args_.options.count(name) > 0; }
  std::string str(const std::string& name, const std::string& def) const {
    const auto it = args_.options.find(name);
    return it == args_.options.end() ? def : it->second;
  }
  std::uint64_t u64(const std::string& name, std::uint64_t def) const {
    const auto it = args_.options.find(name);
    return it == args_.options.end() ? def : parse_u64(it->second, "--" + name);
  }
  std::size_t size(const std::string& name, std::size_t def) const { return static_cast<std::size_t>(u64(name, def)); }

 private:
  const CommandLine::Args& args_;
};

void require_positional(const CommandLine::Args& a, std::size_t min_n, std::size_t max_n, const char* usage) {
  if (a.positional.size() < min_n || a.positional.size() > max_n) {
    throw InvalidArgument(std::string("usage: ") + usage);
  }
}

IndexConfig config_from_options(const Options& o, std::size_t dim) {
  IndexConfig c;
  c.dim = dim;
  c.metric = parse_metric(o.str("metric", "l2"));
  const std::string kind = o.str("index", "hnsw");
  if (kind == "bruteforce" || kind == "flat" || kind == "brute") {
    c.kind = IndexKind::BruteForce;
  } else if (kind == "hnsw" || kind == "nsw") {
    c.kind = IndexKind::HNSW;
    c.hnsw.single_layer = kind == "nsw";
  } else {
    throw InvalidArgument("unknown index type '" + kind + "' (expected hnsw, nsw or bruteforce)");
  }
  c.hnsw.M = o.size("M", 16);
  c.hnsw.M0 = o.size("M0", 0);
  c.hnsw.ef_construction = o.size("efc", 200);
  c.hnsw.seed = o.u64("seed", 42);
  c.hnsw.selection = parse_selection(o.str("select", "heuristic"));
  c.hnsw.delete_policy = parse_delete_policy(o.str("delete-policy", "repair"));
  if (c.kind == IndexKind::HNSW) c.hnsw.validate();
  return c;
}

}  // namespace

// ------------------------------------------------------------------------------------------
CommandLine::Args parse_command_line(const std::string& line) {
  std::vector<std::string> tokens;
  std::string cur;
  bool in_quotes = false;
  bool has_token = false;
  for (const char ch : line) {
    if (ch == '"') {
      in_quotes = !in_quotes;
      has_token = true;
    } else if (!in_quotes && std::isspace(static_cast<unsigned char>(ch))) {
      if (has_token) tokens.push_back(cur);
      cur.clear();
      has_token = false;
    } else {
      cur += ch;
      has_token = true;
    }
  }
  if (in_quotes) throw InvalidArgument("unbalanced double quote");
  if (has_token) tokens.push_back(cur);

  CommandLine::Args a;
  if (tokens.empty()) return a;
  a.command = tokens[0];
  std::transform(a.command.begin(), a.command.end(), a.command.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    const std::string& t = tokens[i];
    if (t.size() > 2 && t[0] == '-' && t[1] == '-') {
      const std::string name = t.substr(2);
      std::string value;
      if (boolean_flags().count(name) == 0 && i + 1 < tokens.size() &&
          !(tokens[i + 1].size() > 2 && tokens[i + 1].compare(0, 2, "--") == 0)) {
        value = tokens[++i];
      }
      if (boolean_flags().count(name) > 0) value = "1";
      if (!a.options.emplace(name, value).second) throw InvalidArgument("option --" + name + " given twice");
    } else {
      a.positional.push_back(t);
    }
  }
  return a;
}

// ------------------------------------------------------------------------------------------
CommandLine::CommandLine(std::ostream& out, std::ostream& err) : out_(out), err_(err) {}

const std::map<std::string, CommandLine::Spec>& CommandLine::commands() {
  static const std::map<std::string, Spec> table = {
      {"help", {&CommandLine::cmd_help, "help [command]", "list commands or show one command's usage"}},
      {"build",
       {&CommandLine::cmd_build,
        "build <file.fvecs|file.csv> [--limit N] [--start-id 0] [--index hnsw|nsw|bruteforce] [--metric l2|ip|cosine] "
        "[--M 16] [--M0 2M] [--efc 200] [--seed 42] [--select heuristic|simple] [--delete-policy repair|repair-prune|repair-fixed|tombstone]",
        "build a new index from a vector file (replaces the current index on success)"}},
      {"new", {&CommandLine::cmd_new, "new <dim> [index options as for build]", "create an empty index"}},
      {"insert",
       {&CommandLine::cmd_insert, "insert <file> [--limit N] [--start-id ID]",
        "add every vector of a file (all-or-nothing)"}},
      {"add", {&CommandLine::cmd_add, "add <id> <v1,v2,...,vd>", "add one vector"}},
      {"upsert", {&CommandLine::cmd_upsert, "upsert <id> <v1,v2,...,vd>", "insert or replace one vector"}},
      {"delete", {&CommandLine::cmd_delete, "delete <id> [<id> ...]", "delete ids (all-or-nothing)"}},
      {"query",
       {&CommandLine::cmd_query,
        "query <file> [--k 10] [--ef 50] [--limit N] [--show 10] [--gt gt.ivecs] [--exact] [--out results.csv]",
        "run k-NN queries from a file; --gt/--exact report recall@k"}},
      {"search", {&CommandLine::cmd_search, "search <v1,v2,...,vd> [--k 10] [--ef 50]", "run one ad-hoc query"}},
      {"save", {&CommandLine::cmd_save, "save <path>", "write the index to a file (atomic)"}},
      {"load", {&CommandLine::cmd_load, "load <path>", "replace the current index with one read from a file"}},
      {"info", {&CommandLine::cmd_info, "info", "one-line summary of the current index"}},
      {"stats", {&CommandLine::cmd_stats, "stats", "graph structure and memory statistics"}},
      {"check", {&CommandLine::cmd_check, "check", "verify all structural invariants"}},
      {"compact", {&CommandLine::cmd_compact, "compact", "rebuild without deleted vectors"}},
      {"set", {&CommandLine::cmd_set, "set <k|ef|timing|echo> <value>", "change a session default"}},
      {"echo", {&CommandLine::cmd_echo, "echo <text>", "print text (useful in scripts)"}},
      {"quit", {&CommandLine::cmd_quit, "quit", "leave the program"}},
      {"exit", {&CommandLine::cmd_quit, "exit", "leave the program"}},
  };
  return table;
}

bool CommandLine::execute(const std::string& line) {
  try {
    const std::string t = trim(line);
    if (t.empty() || t[0] == '#') return true;
    const Args a = parse_command_line(t);
    const auto it = commands().find(a.command);
    if (it == commands().end()) throw InvalidArgument("unknown command '" + a.command + "' (type 'help')");
    (this->*(it->second.handler))(a);
    out_.flush();
    return true;
  } catch (const std::bad_alloc&) {
    out_.flush();
    err_ << "error: out of memory" << std::endl;
  } catch (const std::exception& e) {
    out_.flush();
    err_ << "error: " << e.what() << std::endl;
  }
  return false;
}

int CommandLine::run(std::istream& in, bool interactive, bool keep_going) {
  int code = 0;
  std::string line;
  while (!exit_) {
    if (interactive) out_ << "> " << std::flush;
    if (!std::getline(in, line)) break;
    const std::string t = trim(line);
    if (!interactive && echo_ && !t.empty() && t[0] != '#') out_ << "> " << t << "\n";
    if (!execute(line)) {
      code = 1;
      if (!interactive && !keep_going) break;
    }
  }
  if (interactive) out_ << "\n";
  out_.flush();
  return code;
}

Index& CommandLine::require_index() const {
  if (!index_) throw InvalidArgument("no index loaded (use 'build', 'new' or 'load' first)");
  return *index_;
}

void CommandLine::note_label(label_t id) noexcept {
  if (id >= next_id_) next_id_ = id + 1;
}

void CommandLine::insert_batch(Index& idx, const std::vector<label_t>& ids, const float* data, std::size_t n,
                               std::size_t d) {
  // Validate everything first so that a bad row leaves the index untouched.
  OpenHashMap<label_t, std::uint8_t> seen(n * 2 + 16);
  for (std::size_t i = 0; i < n; ++i) {
    try {
      idx.validate_vector(data + i * d, d);
    } catch (const InvalidArgument& e) {
      throw InvalidArgument("row " + std::to_string(i) + " (id " + std::to_string(ids[i]) + "): " + e.what());
    }
    if (idx.contains(ids[i])) throw InvalidArgument("id " + std::to_string(ids[i]) + " already exists");
    if (!seen.insert(ids[i], 1)) throw InvalidArgument("id " + std::to_string(ids[i]) + " appears twice in the input");
  }
  idx.reserve(idx.size() + n);
  for (std::size_t i = 0; i < n; ++i) {
    idx.add(ids[i], data + i * d, d);
    note_label(ids[i]);
  }
}

void CommandLine::print_results(const std::string& tag, const std::vector<SearchResult>& res, double ms) {
  out_ << tag << ":";
  if (res.empty()) out_ << " (no results)";
  for (const SearchResult& r : res) out_ << " (" << r.id << ", " << fmt_dist(r.distance) << ")";
  if (timing_) out_ << "  " << fmt_fixed(ms, 3) << " ms";
  out_ << "\n";
}

// ------------------------------------------------------------------------------------------
void CommandLine::cmd_help(const Args& a) {
  require_positional(a, 0, 1, "help [command]");
  if (a.positional.size() == 1) {
    const auto it = commands().find(a.positional[0]);
    if (it == commands().end()) throw InvalidArgument("unknown command '" + a.positional[0] + "'");
    out_ << "usage: " << it->second.usage << "\n  " << it->second.summary << "\n";
    return;
  }
  out_ << "commands:\n";
  for (const auto& kv : commands()) {
    if (kv.first == "exit") continue;
    out_ << "  " << std::left << std::setw(8) << kv.first << " " << kv.second.summary << "\n";
  }
  out_ << "type 'help <command>' for its options\n";
}

void CommandLine::cmd_build(const Args& a) {
  require_positional(a, 1, 1, commands().at("build").usage);
  const Options o(a, {"limit", "start-id", "index", "metric", "M", "M0", "efc", "seed", "select", "delete-policy"});
  const LabeledVectors data = read_vector_file(a.positional[0], o.size("limit", kNoLimit), o.u64("start-id", 0));
  if (data.vectors.rows == 0) throw InvalidArgument("'" + a.positional[0] + "' contains no vectors");
  const IndexConfig cfg = config_from_options(o, data.vectors.cols);
  std::unique_ptr<Index> fresh = make_index(cfg);
  const label_t saved_next = next_id_;
  next_id_ = 0;
  const Timer timer;
  try {
    insert_batch(*fresh, data.ids, data.vectors.data.data(), data.vectors.rows, data.vectors.cols);
  } catch (...) {
    next_id_ = saved_next;
    throw;
  }
  const double secs = timer.seconds();
  index_ = std::move(fresh);
  out_ << "built " << index_->describe() << " with " << index_->size() << " vectors";
  if (timing_) out_ << " in " << fmt_fixed(secs, 3) << " s";
  out_ << "\n";
}

void CommandLine::cmd_new(const Args& a) {
  require_positional(a, 1, 1, "new <dim> [index options]");
  const Options o(a, {"index", "metric", "M", "M0", "efc", "seed", "select", "delete-policy"});
  const std::size_t dim = static_cast<std::size_t>(parse_u64(a.positional[0], "dimension"));
  index_ = make_index(config_from_options(o, dim));
  next_id_ = 0;
  out_ << "created empty " << index_->describe() << "\n";
}

void CommandLine::cmd_insert(const Args& a) {
  require_positional(a, 1, 1, commands().at("insert").usage);
  const Options o(a, {"limit", "start-id"});
  Index& idx = require_index();
  const LabeledVectors data = read_vector_file(a.positional[0], o.size("limit", kNoLimit), o.u64("start-id", next_id_));
  if (data.vectors.rows > 0 && data.vectors.cols != idx.dim()) {
    throw InvalidArgument("dimension mismatch: file has " + std::to_string(data.vectors.cols) +
                          "-d vectors, index expects " + std::to_string(idx.dim()));
  }
  const Timer timer;
  insert_batch(idx, data.ids, data.vectors.data.data(), data.vectors.rows, data.vectors.cols);
  out_ << "inserted " << data.vectors.rows << " vectors (live: " << idx.size() << ")";
  if (timing_) out_ << " in " << fmt_fixed(timer.seconds(), 3) << " s";
  out_ << "\n";
}

void CommandLine::cmd_add(const Args& a) {
  require_positional(a, 2, 2, commands().at("add").usage);
  const Options o(a, {});
  Index& idx = require_index();
  const label_t id = parse_u64(a.positional[0], "id");
  const std::vector<float> v = parse_float_list(a.positional[1]);
  idx.add(id, v.data(), v.size());
  note_label(id);
  out_ << "added id " << id << " (live: " << idx.size() << ")\n";
}

void CommandLine::cmd_upsert(const Args& a) {
  require_positional(a, 2, 2, commands().at("upsert").usage);
  const Options o(a, {});
  Index& idx = require_index();
  const label_t id = parse_u64(a.positional[0], "id");
  const std::vector<float> v = parse_float_list(a.positional[1]);
  idx.validate_vector(v.data(), v.size());  // fail before touching the old vector
  const bool existed = idx.contains(id);
  if (existed) idx.remove(id);
  idx.add(id, v.data(), v.size());
  note_label(id);
  out_ << (existed ? "replaced id " : "added id ") << id << " (live: " << idx.size() << ")\n";
}

void CommandLine::cmd_delete(const Args& a) {
  if (a.positional.empty()) throw InvalidArgument(std::string("usage: ") + commands().at("delete").usage);
  const Options o(a, {});
  Index& idx = require_index();
  std::vector<label_t> ids;
  OpenHashMap<label_t, std::uint8_t> seen;
  for (const std::string& s : a.positional) {
    const label_t id = parse_u64(s, "id");
    if (!idx.contains(id)) throw NotFound("id " + std::to_string(id) + " is not in the index");
    if (!seen.insert(id, 1)) throw InvalidArgument("id " + std::to_string(id) + " is listed twice");
    ids.push_back(id);
  }
  const Timer timer;
  for (const label_t id : ids) idx.remove(id);
  out_ << "deleted " << ids.size() << (ids.size() == 1 ? " id" : " ids") << " (live: " << idx.size() << ")";
  if (timing_) out_ << " in " << fmt_fixed(timer.millis(), 3) << " ms";
  out_ << "\n";
}

void CommandLine::cmd_query(const Args& a) {
  require_positional(a, 1, 1, commands().at("query").usage);
  const Options o(a, {"k", "ef", "limit", "show", "gt", "exact", "out"});
  const Index& idx = require_index();
  const std::size_t k = o.size("k", default_k_);
  SearchParams params;
  params.ef = o.size("ef", default_ef_);
  const std::size_t show = o.size("show", 10);
  const LabeledVectors queries = read_vector_file(a.positional[0], o.size("limit", kNoLimit), 0);
  if (queries.vectors.rows > 0 && queries.vectors.cols != idx.dim()) {
    throw InvalidArgument("dimension mismatch: query file has " + std::to_string(queries.vectors.cols) +
                          "-d vectors, index expects " + std::to_string(idx.dim()));
  }
  const std::size_t nq = queries.vectors.rows;
  for (std::size_t i = 0; i < nq; ++i) idx.validate_vector(queries.vectors.row(i), queries.vectors.cols);

  // Optional reference answers for recall@k.
  std::vector<std::vector<label_t>> truth;
  if (o.has("gt") && o.has("exact")) throw InvalidArgument("use either --gt or --exact, not both");
  if (o.has("gt")) {
    const auto rows = read_ivecs(o.str("gt", ""), nq);
    if (rows.size() < nq) throw InvalidArgument("ground-truth file has fewer rows than there are queries");
    for (const auto& r : rows) {
      if (r.size() < k) throw InvalidArgument("ground-truth rows have " + std::to_string(r.size()) + " ids, need k=" + std::to_string(k));
      std::vector<label_t> ids;
      for (std::size_t j = 0; j < k; ++j) ids.push_back(static_cast<label_t>(r[j]));
      truth.push_back(std::move(ids));
    }
  } else if (o.has("exact")) {
    IndexConfig cfg;
    cfg.kind = IndexKind::BruteForce;
    cfg.metric = idx.metric();
    cfg.dim = idx.dim();
    std::unique_ptr<Index> exact = make_index(cfg);
    std::vector<float> v;
    for (const label_t id : idx.labels()) {
      idx.get_vector(id, v);
      exact->add(id, v.data(), v.size());
    }
    SearchParams none;
    for (std::size_t i = 0; i < nq; ++i) {
      std::vector<label_t> ids;
      for (const SearchResult& r : exact->search(queries.vectors.row(i), queries.vectors.cols, k, none, nullptr)) {
        ids.push_back(r.id);
      }
      truth.push_back(std::move(ids));
    }
  }

  std::ofstream out_file;
  if (o.has("out")) {
    out_file.open(o.str("out", ""));
    if (!out_file) throw IoError("cannot open '" + o.str("out", "") + "' for writing");
    out_file << "query,rank,id,distance\n";
    out_file << std::setprecision(9);
  }

  std::vector<double> lat_ms;
  lat_ms.reserve(nq);
  double hits = 0.0;
  double possible = 0.0;
  const Timer total;
  for (std::size_t i = 0; i < nq; ++i) {
    const Timer t;
    const std::vector<SearchResult> res = idx.search(queries.vectors.row(i), queries.vectors.cols, k, params, nullptr);
    const double ms = t.millis();
    lat_ms.push_back(ms);
    if (i < show) print_results("q" + std::to_string(i), res, ms);
    if (out_file.is_open()) {
      for (std::size_t r = 0; r < res.size(); ++r) out_file << i << ',' << r << ',' << res[r].id << ',' << res[r].distance << '\n';
    }
    if (!truth.empty()) {
      const std::vector<label_t>& g = truth[i];
      for (const SearchResult& r : res) hits += std::find(g.begin(), g.end(), r.id) != g.end() ? 1.0 : 0.0;
      possible += static_cast<double>(std::min(k, g.size()));
    }
  }
  const double total_s = total.seconds();
  if (nq > show) out_ << "... (" << (nq - show) << " more queries not shown)\n";
  out_ << "summary: " << nq << " queries, k=" << k << ", ef=" << params.ef;
  if (timing_ && nq > 0) {
    const Summary s = summarize(lat_ms);
    out_ << ", mean " << fmt_fixed(s.mean, 3) << " ms, p99 " << fmt_fixed(s.p99, 3) << " ms, "
         << fmt_fixed(total_s > 0 ? static_cast<double>(nq) / total_s : 0.0, 0) << " QPS";
  }
  if (!truth.empty()) out_ << ", recall@" << k << " " << fmt_fixed(possible > 0 ? hits / possible : 1.0, 4);
  out_ << "\n";
  if (out_file.is_open()) {
    if (!out_file) throw IoError("write to '" + o.str("out", "") + "' failed");
    out_ << "wrote results to " << o.str("out", "") << "\n";
  }
}

void CommandLine::cmd_search(const Args& a) {
  require_positional(a, 1, 1, commands().at("search").usage);
  const Options o(a, {"k", "ef"});
  const Index& idx = require_index();
  const std::vector<float> q = parse_float_list(a.positional[0]);
  SearchParams params;
  params.ef = o.size("ef", default_ef_);
  const Timer t;
  const auto res = idx.search(q.data(), q.size(), o.size("k", default_k_), params, nullptr);
  print_results("result", res, t.millis());
}

void CommandLine::cmd_save(const Args& a) {
  require_positional(a, 1, 1, "save <path>");
  const Options o(a, {});
  const Index& idx = require_index();
  const Timer t;
  save_index(idx, a.positional[0]);
  out_ << "saved " << idx.describe() << " (" << idx.size() << " live vectors) to " << a.positional[0];
  if (timing_) out_ << " in " << fmt_fixed(t.seconds(), 3) << " s";
  out_ << "\n";
}

void CommandLine::cmd_load(const Args& a) {
  require_positional(a, 1, 1, "load <path>");
  const Options o(a, {});
  const Timer t;
  std::unique_ptr<Index> loaded = load_index(a.positional[0]);
  label_t next = 0;
  for (const label_t id : loaded->labels()) next = std::max(next, id + 1);
  index_ = std::move(loaded);
  next_id_ = next;
  out_ << "loaded " << index_->describe() << " (" << index_->size() << " live vectors) from " << a.positional[0];
  if (timing_) out_ << " in " << fmt_fixed(t.seconds(), 3) << " s";
  out_ << "\n";
}

void CommandLine::cmd_info(const Args& a) {
  require_positional(a, 0, 0, "info");
  const Options o(a, {});
  const Index& idx = require_index();
  const IndexStats s = idx.stats();
  out_ << idx.describe() << ": " << s.live << " live vectors";
  if (idx.kind() == IndexKind::HNSW) {
    out_ << ", " << s.tombstones << " tombstones, max level " << s.max_level;
    if (s.has_entry) out_ << ", entry id " << s.entry_id;
  }
  out_ << "\n";
}

void CommandLine::cmd_stats(const Args& a) {
  require_positional(a, 0, 0, "stats");
  const Options o(a, {});
  out_ << format_stats(require_index().stats());
}

void CommandLine::cmd_check(const Args& a) {
  require_positional(a, 0, 0, "check");
  const Options o(a, {});
  const Index& idx = require_index();
  const std::vector<std::string> errs = idx.check_invariants();
  if (!errs.empty()) {
    std::string msg = "invariant check failed:";
    for (const std::string& e : errs) msg += "\n  " + e;
    throw Error(msg);
  }
  out_ << "ok: all invariants hold (" << idx.size() << " live vectors)\n";
}

void CommandLine::cmd_compact(const Args& a) {
  require_positional(a, 0, 0, "compact");
  const Options o(a, {});
  Index& idx = require_index();
  const std::size_t before = idx.stats().tombstones;
  const Timer t;
  idx.compact();
  out_ << "compacted: removed " << before << " tombstones, " << idx.size() << " live vectors";
  if (timing_) out_ << " in " << fmt_fixed(t.seconds(), 3) << " s";
  out_ << "\n";
}

void CommandLine::cmd_set(const Args& a) {
  require_positional(a, 2, 2, "set <k|ef|timing|echo> <value>");
  const Options o(a, {});
  const std::string& key = a.positional[0];
  const std::string& val = a.positional[1];
  auto on_off = [&](const std::string& v) {
    if (v == "on" || v == "1" || v == "true") return true;
    if (v == "off" || v == "0" || v == "false") return false;
    throw InvalidArgument("expected on or off, got '" + v + "'");
  };
  if (key == "k") {
    default_k_ = static_cast<std::size_t>(parse_u64(val, "k"));
  } else if (key == "ef") {
    default_ef_ = static_cast<std::size_t>(parse_u64(val, "ef"));
  } else if (key == "timing") {
    timing_ = on_off(val);
  } else if (key == "echo") {
    echo_ = on_off(val);
  } else {
    throw InvalidArgument("unknown setting '" + key + "' (expected k, ef, timing or echo)");
  }
  out_ << key << " = " << val << "\n";
}

void CommandLine::cmd_echo(const Args& a) {
  std::string text;
  for (std::size_t i = 0; i < a.positional.size(); ++i) text += (i ? " " : "") + a.positional[i];
  for (const auto& kv : a.options) text += (text.empty() ? "--" : " --") + kv.first + (kv.second.empty() ? "" : " " + kv.second);
  out_ << text << "\n";
}

void CommandLine::cmd_quit(const Args& a) {
  require_positional(a, 0, 0, "quit");
  exit_ = true;
}

}  // namespace hnsw
