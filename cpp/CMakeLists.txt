cmake_minimum_required(VERSION 3.16)

project(microstructx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(microstructx_core STATIC
    src/execution_core.cpp
    src/csv_loader.cpp
)

target_include_directories(microstructx_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(microstructx_core PRIVATE /W4 /O2)
else()
    target_compile_options(microstructx_core PRIVATE -Wall -Wextra -Wpedantic -O3 -march=native)
endif()

add_executable(microstructx_bench
    benchmarks/execution_benchmark.cpp
)

target_link_libraries(microstructx_bench PRIVATE microstructx_core)
