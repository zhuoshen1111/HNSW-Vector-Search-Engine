#!/usr/bin/env bash
# Runs the unit tests and the end-to-end golden tests.
#
#   scripts/run_tests.sh [build-dir] [--golden-only | --update-golden]
#
# Golden tests: every tests/golden/<name>.cmds script is run through hnsw_cli --no-timing
# --keep-going; stdout+stderr must equal <name>.expected and the exit code must equal
# <name>.exitcode (default 0). --update-golden rewrites the expected files (review the diff!).
set -u
cd "$(dirname "$0")/.."
BUILD=${1:-build}
MODE=${2:-}
BIN="$BUILD/bin"
mkdir -p tests/tmp
status=0

if [ "$MODE" != "--golden-only" ] && [ "$MODE" != "--update-golden" ]; then
  echo "== unit tests ($BIN/hnsw_tests)"
  if "$BIN/hnsw_tests" 2> tests/tmp/unit.log; then
    tail -n 1 tests/tmp/unit.log
  else
    grep -E "FAILED|FAIL|tests," tests/tmp/unit.log
    status=1
  fi
fi

echo "== golden end-to-end tests ($BIN/hnsw_cli)"
for script in tests/golden/*.cmds; do
  name=$(basename "$script" .cmds)
  expected="tests/golden/$name.expected"
  want_code=0
  [ -f "tests/golden/$name.exitcode" ] && want_code=$(cat "tests/golden/$name.exitcode")
  actual="tests/tmp/$name.actual"
  "$BIN/hnsw_cli" --no-timing --keep-going "$script" > "$actual" 2>&1
  code=$?
  if [ "$MODE" = "--update-golden" ]; then
    cp "$actual" "$expected"
    echo "updated $expected (exit code $code)"
    continue
  fi
  if [ "$code" != "$want_code" ]; then
    echo "FAIL $name: exit code $code, expected $want_code"
    status=1
  elif diff -u "$expected" "$actual" > "tests/tmp/$name.diff"; then
    echo "ok   $name"
  else
    echo "FAIL $name (diff in tests/tmp/$name.diff)"
    cat "tests/tmp/$name.diff"
    status=1
  fi
done

[ $status -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $status
