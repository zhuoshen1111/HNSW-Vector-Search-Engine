#include "hnsw/timer.h"

#include <algorithm>
#include <cmath>

namespace hnsw {

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  p = std::min(100.0, std::max(0.0, p));
  // Nearest-rank definition: smallest value with at least p% of the data at or below it.
  const double rank = std::ceil(p / 100.0 * static_cast<double>(values.size()));
  const std::size_t idx = rank < 1.0 ? 0 : static_cast<std::size_t>(rank) - 1;
  return values[std::min(idx, values.size() - 1)];
}

Summary summarize(const std::vector<double>& values) {
  Summary s;
  s.count = values.size();
  if (values.empty()) return s;
  double sum = 0.0;
  for (double v : values) sum += v;
  s.mean = sum / static_cast<double>(values.size());
  double sq = 0.0;
  for (double v : values) sq += (v - s.mean) * (v - s.mean);
  s.stddev = values.size() > 1 ? std::sqrt(sq / static_cast<double>(values.size() - 1)) : 0.0;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  s.min = sorted.front();
  s.max = sorted.back();
  const std::size_t n = sorted.size();
  s.median = n % 2 == 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  s.p99 = percentile(sorted, 99.0);
  return s;
}

}  // namespace hnsw
