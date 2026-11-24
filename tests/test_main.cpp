// Test runner: ./hnsw_tests [substring-filter]
#include <chrono>
#include <cstring>

#include "test_framework.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int run = 0;
  int failed_tests = 0;
  const auto start = std::chrono::steady_clock::now();
  for (const tf::TestCase& t : tf::registry()) {
    if (filter && std::strstr(t.name, filter) == nullptr) continue;
    ++run;
    const int before = tf::failure_count();
    std::cerr << "[ RUN  ] " << t.name << "\n";
    try {
      t.fn();
    } catch (const tf::AbortTest&) {
      // failure already recorded
    } catch (const std::exception& e) {
      tf::report(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
    } catch (...) {
      tf::report(__FILE__, __LINE__, "unexpected non-standard exception");
    }
    const bool ok = tf::failure_count() == before;
    if (!ok) ++failed_tests;
    std::cerr << (ok ? "[  OK  ] " : "[ FAIL ] ") << t.name << "\n";
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cerr << "\n" << run << " tests, " << failed_tests << " failed, " << tf::failure_count()
            << " failed checks (" << secs << " s)\n";
  return failed_tests == 0 && run > 0 ? 0 : 1;
}
