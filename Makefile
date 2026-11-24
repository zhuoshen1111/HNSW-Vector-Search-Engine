# HNSW Vector Search Engine — primary build (GNU Make + any C++17 compiler).
#
#   make                 optimised build of all binaries into build/bin
#   make test            unit tests + end-to-end golden tests
#   make asan            AddressSanitizer + UBSan build in build-asan/ and run all tests
#   make scalar          library variant with SIMD kernels disabled (build-scalar/)
#   make clean
#
# Variables: CXX, OPT (default -O3), BUILD (default build), SAN (sanitizer flags), EXTRA.

CXX      ?= c++
OPT      ?= -O3
BUILD    ?= build
SAN      ?=
EXTRA    ?=

WARN := -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual \
        -Wcast-align -Wnull-dereference -Wold-style-cast -Wformat=2

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  ARCHFLAGS ?= -march=native
else
  ARCHFLAGS ?=
endif

CXXFLAGS := -std=c++17 $(OPT) $(ARCHFLAGS) $(WARN) $(SAN) $(EXTRA) -Iinclude -MMD -MP
LDFLAGS  += $(SAN) -pthread

LIB_SRC  := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJ  := $(LIB_SRC:%.cpp=$(BUILD)/obj/%.o)
TEST_SRC := $(wildcard tests/*.cpp)
TEST_OBJ := $(TEST_SRC:%.cpp=$(BUILD)/obj/%.o)
LIB      := $(BUILD)/libhnsw.a
BIN      := $(BUILD)/bin

PROGRAMS := $(BIN)/hnsw_cli $(BIN)/hnsw_gen $(BIN)/hnsw_bench $(BIN)/hnsw_tests

.PHONY: all test unit golden asan scalar clean experiments experiments-quick

all: $(PROGRAMS)

$(BUILD)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJ)
	@mkdir -p $(dir $@)
	rm -f $@
	ar rcs $@ $^

$(BIN)/hnsw_cli: $(BUILD)/obj/src/main.o $(LIB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/hnsw_gen: $(BUILD)/obj/tools/gen_data.o $(BUILD)/obj/tools/bench_common.o $(LIB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/hnsw_bench: $(BUILD)/obj/tools/bench.o $(BUILD)/obj/tools/bench_common.o $(LIB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/hnsw_tests: $(TEST_OBJ) $(LIB)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: all
	./scripts/run_tests.sh $(BUILD)

unit: $(BIN)/hnsw_tests
	$(BIN)/hnsw_tests

golden: $(BIN)/hnsw_cli
	./scripts/run_tests.sh $(BUILD) --golden-only

asan:
	$(MAKE) BUILD=build-asan OPT="-O1 -g -fno-omit-frame-pointer" \
	        SAN="-fsanitize=address,undefined -fno-sanitize-recover=undefined" test

scalar:
	$(MAKE) BUILD=build-scalar EXTRA="-DHNSW_FORCE_SCALAR" all

experiments-quick: all
	python3 scripts/run_experiments.py --quick

experiments: all
	python3 scripts/run_experiments.py

clean:
	rm -rf build build-asan build-scalar tests/tmp

-include $(LIB_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(BUILD)/obj/src/main.d $(BUILD)/obj/tools/gen_data.d \
         $(BUILD)/obj/tools/bench.d $(BUILD)/obj/tools/bench_common.d
