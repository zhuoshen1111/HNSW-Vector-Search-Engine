// Wall-clock timing and small summary statistics shared by the CLI and the benchmark driver.
#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace hnsw {

/// Monotonic stopwatch (std::chrono::steady_clock).
class Timer {
 public:
  Timer() : start_(Clock::now()) {}
  void restart() noexcept { start_ = Clock::now(); }
  double seconds() const noexcept { return std::chrono::duration<double>(Clock::now() - start_).count(); }
  double millis() const noexcept { return seconds() * 1e3; }
  double micros() const noexcept { return seconds() * 1e6; }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

struct Summary {
  std::size_t count = 0;
  double mean = 0.0;
  double stddev = 0.0;  ///< sample standard deviation (n - 1); 0 for a single value
  double min = 0.0;
  double median = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

/// Nearest-rank percentile of `values` (p in [0, 100]); 0 for an empty input.
double percentile(std::vector<double> values, double p);

/// Count, mean, sample stddev, min, median, p99 and max of `values`.
Summary summarize(const std::vector<double>& values);

}  // namespace hnsw
