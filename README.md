# Improved Arrow–Hurwicz (IAH)

This repository implements the Improved Arrow–Hurwicz (IAH) method for solving natural convection equations. The project includes accuracy verification in 2D and 3D, and a classical differentially heated cavity benchmark.

A fully self-contained CMake superbuild automatically builds deal.II v9.5.0 and its dependencies from a pinned git submodules and then compiles the IAH executables. No manual installation of deal.II is required.

Clone the repository
```bash 
git clone git@github.com:maggul-research/IAH.git
cd IAH
```

Initialize the deal.II submodule
```bash
git submodule update --init --recursive
```

(Optional) Verify correct version:
```bash
cd external/dealii
git checkout v9.5.0
cd ../../
```

There are three code files to execute:

* ``Accuracy_2d.cc``  -- compares the computed results against a true solution in 2D
* ``Accuracy_3d.cc``  -- compares the computed results against a true solution in 3D
* ``Benchmark.cc`` -- runs differentially heated walls simulation in both 2D and 3D

These are built automatically.

Build with superbuild (recommended)

The superbuild will

1. build deal.II v9.5.0 into build/_deps/dealii_install

2. build the IAH executables into build/iah-build

To build

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target all-build -j"$(nproc)"
```

Executables appear in

```bash
build/iah-build/
```

Run them with

```bash
./build/iah-build/Accuracy_2d <alg_number:1-3> <num_refinements:2-9>
./build/iah-build/Accuracy_3d <alg_number:1-3> <num_refinements:2-9>
./build/iah-build/Benchmark
```
