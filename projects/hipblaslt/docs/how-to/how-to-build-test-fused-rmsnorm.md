# Build and Test Fused RMSNorm E2E

This guide summarizes the end-to-end workflow for building hipBLASLt, generating
the gfx950 fused RMSNorm device libraries, and running the focused GoogleTest
coverage:

```bash
./clients/hipblaslt-test --gtest_filter='FusedEpilogue*.*'
```

The flow is useful when validating the composable fused-epilogue RMSNorm path,
including the full RMSNorm and residual-add plus RMSNorm E2E tests.

## Prerequisites

Use a ROCm SDK that contains `amdclang++`, HIP, amd-smi, hipBLAS common, and the
gfx950 runtime libraries. In TheRock-style environments, set:

```bash
export ROCM_PATH=/home/ossci/therock-tarball/install
export ROCM_HOME="$ROCM_PATH"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
```

Confirm that the target GPU is visible:

```bash
rocm_agent_enumerator
rocminfo
```

The expected target for these tests is `gfx950`.

## Python Environment

TensileLite uses Python tools during device-library generation. Create a local
environment and install the TensileLite requirements:

```bash
cd /path/to/rocm-libraries/projects/hipblaslt
python3 -m venv build/python-venv
build/python-venv/bin/python -m pip install --upgrade pip
build/python-venv/bin/python -m pip install -r tensilelite/requirements.txt
```

Configure CMake with this interpreter:

```bash
-DPython_EXECUTABLE=$PWD/build/python-venv/bin/python
-DPython3_EXECUTABLE=$PWD/build/python-venv/bin/python
```

## Compiler Wrapper

Some ROCm clang builds need an explicit GCC install directory to find the host
C++ standard library. The rocisa assembler capability probes also fail if the
compiler prints warnings to stdout or stderr. Use a quiet wrapper:

```bash
mkdir -p build/toolchain
cat > build/toolchain/amdclang++ <<'SH'
#!/usr/bin/env bash
has_assembler=0
prev=
for arg in "$@"; do
  if [ "$prev" = "-x" ] && [ "$arg" = "assembler" ]; then
    has_assembler=1
    break
  fi
  prev="$arg"
done
if [ "$has_assembler" = 1 ]; then
  exec /home/ossci/therock-tarball/install/bin/amdclang++ -Wno-gcc-install-dir-libstdcxx "$@"
fi
exec ccache /home/ossci/therock-tarball/install/bin/amdclang++ \
  -Wno-gcc-install-dir-libstdcxx \
  --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 "$@"
SH
chmod +x build/toolchain/amdclang++
```

Verify that rocisa recognizes gfx950 as supported:

```bash
PYTHONPATH="$PWD/build/release/tensilelite/rocisa:$PWD/build/release/tensilelite:$PWD/tensilelite" \
build/python-venv/bin/python - <<'PY'
from Tensile.Common.Types import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
compiler = "build/toolchain/amdclang++"
info = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)[IsaVersion(9, 5, 0)]
print(info.asmCaps["SupportedISA"])
PY
```

The output should be `1`.

## Local Build Dependencies

If system GTest, BLAS, and LAPACK are unavailable, build the local dependency
bundle:

```bash
mkdir -p build/deps
cmake -S deps -B build/deps \
  -D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -D CMAKE_INSTALL_PREFIX=$PWD/build/deps/install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D BUILD_LAPACK=ON \
  -D BUILD_GTEST=ON
cmake --build build/deps --target googletest lapack --parallel 16
cmake --build build/deps --target install --parallel 16
```

Build msgpack-c into the same prefix for msgpack-mode libraries:

```bash
git clone -b cpp-3.1.0 https://github.com/msgpack/msgpack-c.git build/deps/msgpack-c --depth 1
cmake -S build/deps/msgpack-c -B build/deps/msgpack-c-build \
  -D CMAKE_INSTALL_PREFIX=$PWD/build/deps/install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D MSGPACK_BUILD_TESTS=OFF \
  -D MSGPACK_BUILD_EXAMPLES=OFF \
  -D MSGPACK_CXX17=ON
cmake --build build/deps/msgpack-c-build --target install --parallel 16
```

## Configure hipBLASLt

Configure a focused gfx950 build:

```bash
cmake -S . -B build/release -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=$PWD/build/toolchain/amdclang++ \
  -D CMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache \
  -D CMAKE_PREFIX_PATH="$PWD/build/deps/install;$ROCM_PATH" \
  -D CMAKE_INSTALL_PREFIX=$PWD/hipblaslt-install \
  -D CMAKE_INSTALL_LIBDIR=lib \
  -D ROCM_PATH=$ROCM_PATH \
  -D GPU_TARGETS=gfx950 \
  -D HIPBLASLT_ENABLE_FETCH=ON \
  -D HIPBLASLT_ENABLE_ROCROLLER=OFF \
  -D HIPBLASLT_ENABLE_YAML=OFF \
  -D HIPBLASLT_ENABLE_CLIENT=ON \
  -D HIPBLASLT_BUILD_TESTING=ON \
  -D HIPBLASLT_ENABLE_SAMPLES=OFF \
  -D HIPBLASLT_ENABLE_BLIS=OFF \
  -D HIPBLASLT_ENABLE_MXDATAGENERATOR=ON \
  -D TENSILELITE_ENABLE_CLIENT=ON \
  -D BLAS_LIBRARIES=$PWD/build/deps/install/lib/libblas.a \
  "-D LAPACK_LIBRARIES=$PWD/build/deps/install/lib/liblapack.a;$PWD/build/deps/install/lib/libcblas.a;/usr/lib/gcc/x86_64-linux-gnu/13/libgfortran.so;/usr/lib/gcc/x86_64-linux-gnu/13/libquadmath.so" \
  -D Python_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D Python3_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D CLANG_TIDY_EXE=/bin/false \
  -D RUN_CLANG_TIDY_EXE=/bin/false
```

## Build the Test Binary and Client

Build the GoogleTest binary:

```bash
cmake --build build/release --target hipblaslt-test --parallel 16
```

Build the TensileLite benchmark client. The client is needed to generate
benchmark-derived PartialRMS `3_LibraryLogic` files:

```bash
cmake --build build/release --target tensilelite-client --parallel 16
```

## Generate PartialRMS Library Logic

The full RMSNorm E2E tests need PartialRMS logic that is not part of the normal
generic gfx950 logic path. Generate it from the row-major PartialRMS benchmark
YAML:

```bash
rm -rf /tmp/hipblaslt_partialrms_out
PYTHONPATH="$PWD/build/release/tensilelite/rocisa:$PWD/build/release/tensilelite:$PWD/tensilelite" \
LD_LIBRARY_PATH="$PWD/build/release/tensilelite:$PWD/build/release/clients/common:$PWD/build/release/library:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
build/python-venv/bin/python tensilelite/Tensile/bin/Tensile \
  tensilelite/epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml \
  /tmp/hipblaslt_partialrms_out \
  --cxx-compiler "$PWD/build/toolchain/amdclang++" \
  --gpu-targets gfx950 \
  --prebuilt-client "$PWD/build/release/tensilelite/client/tensilelite-client" \
  --global-parameters LibraryFormat='"msgpack"'
```

This should produce:

```text
/tmp/hipblaslt_partialrms_out/3_LibraryLogic/
  partialrms_k1_Cijk_Alik_Bljk_BBS_BH_PRMS_UserArgs.yaml
  partialrms_k1_Cijk_Alik_Bljk_BBS_BH_PRMS_RA_UserArgs.yaml
```

If the generated files contain `Device 74a1` in the device list, remove it. The
gfx950 chip-ID validator rejects `74a1` because it is not a gfx950 device ID.

## Build Device Libraries

Build the PartialRMS K1 library from the generated logic:

```bash
rm -f build/release/device-library/tensilelite-device-libraries.stamp
rm -f build/release/device-library/tensilelite-device-libraries-TensileLogic.stamp

cmake -S . -B build/release \
  -D HIPBLASLT_LIBLOGIC_PATH=/tmp/hipblaslt_partialrms_out/3_LibraryLogic \
  -D HIPBLASLT_ENABLE_MXDATAGENERATOR=ON \
  -D HIPBLASLT_ENABLE_ROCROLLER=OFF \
  -D HIPBLASLT_ENABLE_YAML=OFF \
  -D Python_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D Python3_EXECUTABLE=$PWD/build/python-venv/bin/python \
  -D CLANG_TIDY_EXE=/bin/false \
  -D RUN_CLANG_TIDY_EXE=/bin/false

cmake --build build/release --target device-library/tensilelite-device-libraries --parallel 16
```

Build and install the row-major Kernel 2 code object:

```bash
cmake --build build/release --target row-div-library-gfx950 --parallel 16
```

The runtime should now have these artifacts:

```text
build/release/Tensile/library/gfx950/
  TensileLibrary_lazy_gfx950.dat.zlib
  TensileLiteLibrary_lazy_gfx950_Mapping.dat.zlib
  row_div_gfx950.co
  ...
```

## Run the Focused Tests

Run the filtered gtest from the build directory:

```bash
cd build/release
LD_LIBRARY_PATH="$PWD/tensilelite:$PWD/clients/common:$PWD/library:$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}" \
./clients/hipblaslt-test --gtest_filter='FusedEpilogue*.*'
```

Expected result:

```text
[==========] 45 tests from 4 test suites ran.
[  PASSED  ] 45 tests.
```

## Troubleshooting

- **`Could not find standard C++ header 'cmath'`**: use the compiler wrapper
  above so HIP/C++ compilation receives `--gcc-install-dir`.
- **`SupportedISA == 0` for gfx950**: ensure the wrapper suppresses
  `-Wgcc-install-dir-libstdcxx` for rocisa assembler probes.
- **`no PartialRMS solution selected`**: the gfx950 library is missing the
  `PRMS`/`PRMS_RA` logic. Regenerate `3_LibraryLogic` and rebuild
  `tensilelite-device-libraries` with `HIPBLASLT_LIBLOGIC_PATH` pointing to it.
- **`getKernel failed: row_div`**: build `row-div-library-gfx950` and make sure
  `row_div_gfx950.co` is present under `build/release/Tensile/library/gfx950`.
- **Residual and non-residual tests interfere with each other**: ensure
  `ContractionProblemGemm` comparison and hashing include the PartialRMS
  discriminator fields (`usePartialRMS`, `partialRMSResidualAdd`,
  `partialRMSMT0`, `partialRMSMT1`, and `useRstdScale`) so solution-cache keys
  do not alias.
