cmake_minimum_required(VERSION 3.18)
project(superbuild-IAH LANGUAGES CXX)

include(ExternalProject)
set(SUPERBUILD_PREFIX ${CMAKE_BINARY_DIR}/_deps)
set(OPENBLAS_PREFIX   ${SUPERBUILD_PREFIX}/openblas)
set(OPENBLAS_INSTALL  ${OPENBLAS_PREFIX}/install)
set(DEALII_PREFIX     ${SUPERBUILD_PREFIX}/dealii)
set(DEALII_INSTALL    ${SUPERBUILD_PREFIX}/dealii_install)

# Build & install OpenBLAS
ExternalProject_Add(openblas_ep
  SOURCE_DIR          ${CMAKE_SOURCE_DIR}/external/OpenBLAS
  GIT_TAG             v0.3.26
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL}
    -DBUILD_SHARED_LIBS=ON
    -DDYNAMIC_ARCH=ON         # portable CPU kernels
    -DUSE_OPENMP=OFF          # deterministic single-threaded by default
    -DUSE_THREAD=OFF          # (OpenBLAS internal pthread-based threading)
  # If your toolchain lacks Fortran, you can try NOFORTRAN=ON,
  # but LAPACK routines require Fortran; recommended is to keep Fortran ON.
)


# Build & install deal.II from the submodule
ExternalProject_Add(dealii_ep
  SOURCE_DIR          ${CMAKE_SOURCE_DIR}/external/dealii
  GIT_TAG             v9.5.0
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${DEALII_INSTALL}
    -DDEAL_II_WITH_LAPACK=ON
    -DDEAL_II_WITH_UMFPACK=ON
    -DLAPACK_LIBRARIES=${OPENBLAS_INSTALL}/lib/libopenblas${CMAKE_SHARED_LIBRARY_SUFFIX}
    -DBLAS_LIBRARIES=${OPENBLAS_INSTALL}/lib/libopenblas${CMAKE_SHARED_LIBRARY_SUFFIX}
    -DCMAKE_PREFIX_PATH=${OPENBLAS_INSTALL}

    -DCMAKE_BUILD_TYPE=Release

    # turn OFF extras
    -DDEAL_II_COMPONENT_EXAMPLES=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION=OFF

    # avoid the old bundled TBB that breaks with newer compilers
    -DDEAL_II_WITH_THREADS=OFF
    -DDEAL_II_WITH_TBB=OFF
    -DDEAL_II_FORCE_BUNDLED_TBB=OFF
    
  BUILD_BYPRODUCTS
    ${DEALII_INSTALL}/lib/cmake/deal.II/deal.IIConfig.cmake
  DEPENDS openblas_ep
)

# Configure the actual project only after deal.II is installed
ExternalProject_Add(iah_ep
  SOURCE_DIR          ${CMAKE_SOURCE_DIR}/src
  BINARY_DIR          ${CMAKE_BINARY_DIR}/iah-build
  LIST_SEPARATOR ::
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DDEAL_II_DIR=${DEALII_INSTALL}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/iah-install
  DEPENDS dealii_ep
)

add_custom_target(all-build DEPENDS iah_ep)
