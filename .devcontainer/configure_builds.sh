#!/bin/bash
#
# Configures the build directories when the devcontainer is created.
#
# Mirrors the cmake invocation of buildstream/dev_shell_bashrc, so that a build
# started here and one started in the BuildStream dev shell are configured
# alike. What the dev shell does on top of it -- ssh server, waypipe tunnel,
# discovery of the host NVIDIA drivers -- is plumbing that exists to reach into
# a sandbox, and has no purpose in a container the editor attaches to directly.
#
# Only Clang/RelWithDebInfo is configured by default, which is what the project
# builds with; dev_shell_bashrc configures the whole Clang/GCC x Debug/
# RelWithDebInfo matrix instead, but four cmake runs is a slow way to start a
# container. Pass the combinations you want to configure to get the others:
#   .devcontainer/configure_builds.sh Clang/Debug GCC/RelWithDebInfo

set -e

SOURCE_DIR="${SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SOURCE_DIR"

declare -A COMPILERS=( [Clang]="$TOOLCHAIN_DIR/clang++" [GCC]="$TOOLCHAIN_DIR/g++" )

# No OpenCL setup here, on purpose. The sysroot carries PortableCL, but built
# without ICD support -- it exports no clIcdGetPlatformIDsKHR and its soname is
# libOpenCL.so.2 -- while the binary links the ICD loader, libOpenCL.so.1. No
# vendor file can bridge those two, so the container has no OpenCL platform to
# offer and FORCE_CPU, which devcontainer.json sets, is what the views fall back
# on. That matches how the test suite is run.

for combination in "${@:-Clang/RelWithDebInfo}"; do
    compiler="${combination%%/*}"
    build_type="${combination##*/}"
    cxx="${COMPILERS[$compiler]}"
    if [ -z "$cxx" ]; then
        echo >&2 "Unknown compiler '$compiler', expected one of: ${!COMPILERS[*]}"
        exit 1
    fi
    build_folder="builds/$TARGET_TRIPLE/$compiler/$build_type"
    if [ -d "$build_folder" ]; then
        echo "$build_folder is already configured, leaving it alone."
        continue
    fi
    echo "Configuring $build_folder with $cxx"
    cmake -Ssrc -B"$build_folder" \
        -DCMAKE_CXX_COMPILER="$cxx" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
done

# clangd looks for the compilation database at the root of the source tree it is
# given, and src/ is where the top level CMakeLists.txt lives.
first_build_folder="builds/$TARGET_TRIPLE/Clang/RelWithDebInfo"
if [ -f "$first_build_folder/compile_commands.json" ] && [ ! -e src/compile_commands.json ]; then
    ln -s "../$first_build_folder/compile_commands.json" src/compile_commands.json
fi

cat << 'EOF'

Ready. Build with:
  cmake --build builds/x86_64-linux-gnu/Clang/RelWithDebInfo
Build and run the test suite with:
  cmake --build builds/x86_64-linux-gnu/Clang/RelWithDebInfo --target squey_run_testsuite
EOF
