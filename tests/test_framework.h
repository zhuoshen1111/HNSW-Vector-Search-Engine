// Minimal self-contained unit-test framework (no external dependency).
//
//   TEST(name) { ... }                  registers a test case
//   CHECK(cond) / CHECK_EQ(a, b)         record a failure and continue
//   REQUIRE(cond)                        record a failure and abort the current test
//   CHECK_THROWS_AS(expr, ExceptionType) expect a specific exception type
//
// The runner (test_main.cpp) executes every registered test, optionally filtered by a substring
// given on the command line, prints a summary and exits non-zero if anything failed.
#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct AbortTest {};

inline int& failure_count() {
  static int n = 0;
  return n;
}

inline void report(const char* file, int line, const std::string& msg) {
  ++failure_count();
  std::cerr << "    FAILED " << file << ":" << line << ": " << msg << "\n";
}

template <typename T, typename = void>
struct is_streamable : std::false_type {};
template <typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>
    : std::true_type {};

template <typename T>
void print_value(std::ostream& os, const T& v) {
  if constexpr (std::is_enum<T>::value) {
    os << static_cast<long long>(v);
  } else if constexpr (is_streamable<T>::value) {
    os << v;
  } else {
    os << "<unprintable>";
  }
}

template <typename A, typename B>
std::string describe_pair(const A& a, const B& b) {
  std::ostringstream os;
  os << "(";
  print_value(os, a);
  os << ") vs (";
  print_value(os, b);
  os << ")";
  return os.str();
}

}  // namespace tf

#define TF_CONCAT2(a, b) a##b
#define TF_CONCAT(a, b) TF_CONCAT2(a, b)

#define TEST(name)                                                    \
  static void name();                                                 \
  static const ::tf::Registrar TF_CONCAT(tf_reg_, name)(#name, &name); \
  static void name()

#define CHECK(cond)                                             \
  do {                                                          \
    if (!(cond)) ::tf::report(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define REQUIRE(cond)                                                  \
  do {                                                                 \
    if (!(cond)) {                                                     \
      ::tf::report(__FILE__, __LINE__, "REQUIRE(" #cond ")");          \
      throw ::tf::AbortTest{};                                         \
    }                                                                  \
  } while (0)

#define CHECK_EQ(a, b)                                                                              \
  do {                                                                                              \
    const auto tf_a = (a); /* copies: operands may refer into temporaries */                       \
    const auto tf_b = (b);                                                                          \
    if (!(tf_a == tf_b)) {                                                                          \
      ::tf::report(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ") " + ::tf::describe_pair(tf_a, tf_b)); \
    }                                                                                               \
  } while (0)

#define CHECK_THROWS_AS(expr, Exc)                                                         \
  do {                                                                                     \
    bool tf_caught = false;                                                                \
    try {                                                                                  \
      (void)(expr);                                                                        \
    } catch (const Exc&) {                                                                 \
      tf_caught = true;                                                                    \
    } catch (const std::exception& tf_e) {                                                 \
      ::tf::report(__FILE__, __LINE__, std::string("wrong exception from " #expr ": ") + tf_e.what()); \
      tf_caught = true;                                                                    \
    }                                                                                      \
    if (!tf_caught) ::tf::report(__FILE__, __LINE__, "expected " #Exc " from " #expr);     \
  } while (0)
