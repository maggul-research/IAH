cmake_minimum_required(VERSION 3.15)

project(RunIAH LANGUAGES CXX)

# Recommended for reproducibility/IDE tooling
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find deal.II
find_package(deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ${CMAKE_PREFIX_PATH}
        $ENV{DEAL_II_DIR}
        ${DEAL_II_DIR}/lib/cmake/deal.II
        ${DEAL_II_DIR}/share/deal.II
)

if(NOT deal.II_FOUND)
  message(FATAL_ERROR "\n*** Could not locate deal.II (>= 9.5.0). ***\n\n"
    "Pass -DDEAL_II_DIR=/path/to/dealii/install or set env DEAL_II_DIR.")
endif()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

set(PROGRAMS
  Accuracy_2d
  Accuracy_3d
  Benchmark
)

foreach(name IN LISTS PROGRAMS)
  set(src "${name}.cc")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    message(FATAL_ERROR "Expected source file not found: ${src}")
  endif()

  add_executable(${name} ${src})
  DEAL_II_SETUP_TARGET(${name})

  # Install so artifacts land in a predictable place
  install(TARGETS ${name} RUNTIME DESTINATION bin)
endforeach()