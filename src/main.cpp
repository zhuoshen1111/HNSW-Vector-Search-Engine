// hnsw_cli: interactive shell / script runner for the vector search engine.
//
//   hnsw_cli                         interactive prompt (when stdin is a terminal)
//   hnsw_cli script.cmds             run a command script, stop at the first error
//   hnsw_cli --keep-going script     run every command, exit 1 if any failed
//   hnsw_cli < script.cmds           read commands from stdin
//
// Exit codes: 0 success, 1 a command failed, 2 bad program arguments.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "hnsw/cli.h"

#if defined(_WIN32)
#include <io.h>
#define HNSW_ISATTY _isatty
#define HNSW_FILENO _fileno
#else
#include <unistd.h>
#define HNSW_ISATTY isatty
#define HNSW_FILENO fileno
#endif

namespace {
void usage(std::ostream& os) {
  os << "usage: hnsw_cli [--keep-going] [--no-timing] [--no-echo] [script]\n"
        "  --keep-going  continue after a failing command (exit code is still 1)\n"
        "  --no-timing   omit wall-clock times (deterministic output for tests)\n"
        "  --no-echo     do not echo script commands\n";
}
}  // namespace

int main(int argc, char** argv) {
  bool keep_going = false;
  bool timing = true;
  bool echo = true;
  std::string script;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--keep-going") {
      keep_going = true;
    } else if (arg == "--no-timing") {
      timing = false;
    } else if (arg == "--no-echo") {
      echo = false;
    } else if (arg == "-h" || arg == "--help") {
      usage(std::cout);
      return 0;
    } else if (!arg.empty() && arg[0] == '-') {
      std::cerr << "error: unknown option '" << arg << "'\n";
      usage(std::cerr);
      return 2;
    } else if (script.empty()) {
      script = arg;
    } else {
      std::cerr << "error: only one script file may be given\n";
      return 2;
    }
  }

  hnsw::CommandLine cli(std::cout, std::cerr);
  cli.set_timing(timing);
  cli.set_echo(echo);
  if (!script.empty()) {
    std::ifstream in(script);
    if (!in) {
      std::cerr << "error: cannot open script '" << script << "'\n";
      return 2;
    }
    return cli.run(in, false, keep_going);
  }
  const bool interactive = HNSW_ISATTY(HNSW_FILENO(stdin)) != 0;
  if (interactive) std::cout << "HNSW vector search engine. Type 'help' for commands, 'quit' to leave.\n";
  return cli.run(std::cin, interactive, keep_going);
}
