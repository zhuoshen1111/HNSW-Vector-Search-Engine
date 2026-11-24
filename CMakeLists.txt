# Alternative build with CMake (the Makefile is the primary, verified build).
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   ctest --test-dir build --output-on-failure
#
# Options: -DHNSW_SANITIZE=ON (ASan+UBSan), -DHNSW_FORCE_SCALAR=ON (no SIMD kernels),
#          -DHNSW_NATIVE=OFF (no -march=native on x86-64).
cmake_minimum_required(VERSION 3.16)
project(hnsw_vector_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(HNSW_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(HNSW_FORCE_SCALAR "Use the scalar reference distance kernels only" OFF)
option(HNSW_NATIVE "Compile with -march=native on x86-64" ON)

find_package(Threads REQUIRED)

file(GLOB HNSW_LIB_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM HNSW_LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_library(hnsw STATIC ${HNSW_LIB_SOURCES})
target_include_directories(hnsw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(hnsw PUBLIC /W4 /permissive-)
else()
  target_compile_options(hnsw PUBLIC -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Woverloaded-virtual
                                     -Wcast-align -Wnull-dereference -Wold-style-cast -Wformat=2)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(hnsw PUBLIC -O3)
  endif()
endif()
if(HNSW_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_compile_options(hnsw PUBLIC -march=native)
endif()
if(HNSW_FORCE_SCALAR)
  target_compile_definitions(hnsw PUBLIC HNSW_FORCE_SCALAR)
endif()
if(HNSW_SANITIZE)
  target_compile_options(hnsw PUBLIC -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer -g)
  target_link_options(hnsw PUBLIC -fsanitize=address,undefined)
endif()
target_link_libraries(hnsw PUBLIC Threads::Threads)

add_executable(hnsw_cli src/main.cpp)
target_link_libraries(hnsw_cli PRIVATE hnsw)

add_executable(hnsw_gen tools/gen_data.cpp tools/bench_common.cpp)
target_link_libraries(hnsw_gen PRIVATE hnsw)

add_executable(hnsw_bench tools/bench.cpp tools/bench_common.cpp)
target_link_libraries(hnsw_bench PRIVATE hnsw)

file(GLOB HNSW_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
add_executable(hnsw_tests ${HNSW_TEST_SOURCES})
target_include_directories(hnsw_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(hnsw_tests PRIVATE hnsw)

enable_testing()
add_test(NAME unit COMMAND hnsw_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME golden
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_tests.sh ${CMAKE_BINARY_DIR} --golden-only
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
