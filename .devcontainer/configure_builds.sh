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

# No OpenCL setup here, on purpose. The sysroot does carry a PortableCL that
# registers itself as an ICD -- libpocl.so.2 alongside its vendor file -- so the
# container has a CPU device to offer, and run_cmd.sh is what points the loader
# at it when the application starts -- and at the GPU drivers too: NVIDIA's
# when prepare_gpu.sh found one on the host, Mesa's for an AMD or Intel GPU.

# Under podman the workspace is the container user's only with --userns=keep-id,
# which the devcontainer CLI passes by itself and Zed only when told it drives
# podman. Without it the workspace is root's in here, and the first cmake fails
# on a directory it cannot create, which says nothing about why. Say it here
# instead, and let the container start anyway, as below.
if [ ! -w . ]; then
    echo >&2 "Not configuring anything: $(id -un) cannot write to $SOURCE_DIR."
    echo >&2 "Under podman the container needs --userns=keep-id. Zed passes it once its"
    echo >&2 "settings say \"use_podman\": true; rebuild the container then."
    exit 0
fi

# A clone without --recursive leaves the submodules empty, and cmake then fails
# with "does not contain a CMakeLists.txt file" for each of them, under a couple
# of hundred lines of consequences -- while the editor reports only that the
# container's scripts failed, which points nowhere. Say it here instead, and let
# the container start anyway: a shell to run the fix in is more use than a
# container that refuses to open.
mapfile -t SUBMODULES < <(sed -n 's/^[[:space:]]*path = //p' .gitmodules 2>/dev/null)
EMPTY=()
for submodule in "${SUBMODULES[@]}"; do
    [ -n "$(ls -A "$submodule" 2>/dev/null)" ] || EMPTY+=("$submodule")
done
if [ ${#EMPTY[@]} -gt 0 ]; then
    echo >&2 "Not configuring anything: these submodules are empty."
    printf >&2 '  %s\n' "${EMPTY[@]}"
    echo >&2 "Populate them, then run this script again:"
    echo >&2 "  git submodule update --init --recursive"
    echo >&2 "  .devcontainer/configure_builds.sh"
    exit 0
fi

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
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
done

# clangd looks for the compilation database at the root of the source tree it is
# given, and src/ is where the top level CMakeLists.txt lives. Point it at the
# database buildstream/clangd.sh gives clangd rather than at the one CMake
# writes, which lists the unity sources instead of the files they include and
# names a precompiled header clangd cannot read. clangd.sh rewrites it whenever
# CMake rewrites its own; a clangd started otherwise sees the last rewrite.
first_build_folder="builds/$TARGET_TRIPLE/Clang/RelWithDebInfo"
if [ -f "$first_build_folder/compile_commands.json" ]; then
    python3 buildstream/scripts/clangd_compile_commands.py \
        "$first_build_folder/compile_commands.json" "$first_build_folder/clangd/compile_commands.json"
    if [ -L src/compile_commands.json ] || [ ! -e src/compile_commands.json ]; then
        ln -sfn "../$first_build_folder/clangd/compile_commands.json" src/compile_commands.json
    fi
fi

cat << 'EOF'

Ready. Build with:
  cmake --build builds/x86_64-linux-gnu/Clang/RelWithDebInfo
Build and run the test suite with:
  cmake --build builds/x86_64-linux-gnu/Clang/RelWithDebInfo --target squey_run_testsuite
EOF
