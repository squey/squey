#!/bin/bash

# clangd for editors and coding assistants.
#
# The compile commands of Squey name headers (/app/include...) that only exist
# inside the development sandbox that buildstream/dev_shell.sh starts, so
# clangd runs there: directly when this script already runs inside it (or
# inside the devcontainer), through the sandbox's ssh server otherwise. The
# sandbox sees the host filesystem at the same paths, which spares translating
# the file URIs exchanged with the client. Outside a Squey checkout, this is
# the host clangd.
#
# Each checkout follows its own build tree, so that sessions working on
# different worktrees never see each other's sources.
#
# SQUEY_SANDBOX_SSH       ssh destination of the Linux sandbox (default: SqueyLinux,
#                         see buildstream/sshd/ssh_config.squey)
# SQUEY_SANDBOX_SSH_OPTS  extra ssh options, e.g. "-o IdentityFile=~/.ssh/<key>"
# SQUEY_CLANGD_BUILD_DIR  build tree to follow, absolute or relative to the
#                         checkout (default: the first configured one of BUILD_DIRS)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The layout of buildstream/dev_shell.sh, then the one of the CMake presets
BUILD_DIRS=(builds/x86_64-linux-gnu/Clang/RelWithDebInfo builds/linux-release)

# Claude Code names the project of the session, editors start the server in it.
# From a submodule, go up to the checkout that holds it.
SOURCE_DIR=$(git -C "${CLAUDE_PROJECT_DIR:-$PWD}" rev-parse --show-superproject-working-tree --show-toplevel 2> /dev/null | head -n 1)
if [ ! -f "$SOURCE_DIR/buildstream/dev_shell.sh" ]; then
    exec clangd "$@"
fi

fail() {
    echo "squey-clangd: $*" >&2
    exit 1
}

# Runs a shell command line where the headers of the build are
if [ -d /app/include/qt6 ]; then
    SANDBOX=(bash -c)
else
    # stdout carries the protocol, and nobody is there to answer a prompt. The
    # forwardings of the host entry (waypipe) are meant for interactive sessions.
    read -r -a SSH_OPTS <<< "$SQUEY_SANDBOX_SSH_OPTS"
    SANDBOX=(ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o LogLevel=ERROR "${SSH_OPTS[@]}"
             "${SQUEY_SANDBOX_SSH:-SqueyLinux}")
    if ! error=$("${SANDBOX[@]}" true 2>&1 < /dev/null > /dev/null); then
        fail "cannot reach the Linux sandbox (${error:-ssh failed}), is buildstream/dev_shell.sh running?"
    fi
fi

# A tree that already has its compile commands, else any configured one
find_build_dir() {
    local candidates=("${BUILD_DIRS[@]}") file dir
    [ -n "$SQUEY_CLANGD_BUILD_DIR" ] && candidates=("$SQUEY_CLANGD_BUILD_DIR")
    for file in compile_commands.json CMakeCache.txt; do
        for dir in "${candidates[@]}"; do
            [[ "$dir" == /* ]] || dir="$SOURCE_DIR/$dir"
            if [ -f "$dir/$file" ]; then
                echo "$dir"
                return
            fi
        done
    done
}

BUILD_DIR=$(find_build_dir)
[ -n "$BUILD_DIR" ] || fail "no configured build tree in $SOURCE_DIR (looked for ${SQUEY_CLANGD_BUILD_DIR:-${BUILD_DIRS[*]}})"
COMPILE_COMMANDS="$BUILD_DIR/compile_commands.json"
CLANGD_DIR="$BUILD_DIR/clangd"

# Trees configured before CMakeLists.txt asked for the compile commands. An ssh
# session lacks the pkg-config path the sandbox shell sets.
if [ ! -f "$COMPILE_COMMANDS" ]; then
    "${SANDBOX[@]}" "export PKG_CONFIG_PATH=\${PKG_CONFIG_PATH:-/app/lib/pkgconfig:} && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $(printf %q "$BUILD_DIR")" < /dev/null >&2
    [ -f "$COMPILE_COMMANDS" ] || fail "$BUILD_DIR still has no compile_commands.json"
fi

# Unity builds and precompiled headers leave compile commands clangd cannot use as is
refresh_compile_commands() {
    if [ "$COMPILE_COMMANDS" -nt "$CLANGD_DIR/compile_commands.json" ]; then
        python3 "$HERE/scripts/clangd_compile_commands.py" "$COMPILE_COMMANDS" "$CLANGD_DIR/compile_commands.json"
    fi
}
refresh_compile_commands >&2 || fail "cannot rewrite $COMPILE_COMMANDS"

# Follow the reconfigurations of the build tree for as long as clangd runs, since
# clangd reloads the compile commands once they change. The protocol streams stay
# out of this loop, so that the client sees them close as soon as clangd exits.
(
    while kill -0 $$ 2> /dev/null; do
        sleep 10
        refresh_compile_commands
    done
) < /dev/null > /dev/null 2>&1 &

ARGS=""
[ $# -gt 0 ] && ARGS=$(printf ' %q' "$@")
exec "${SANDBOX[@]}" "cd $(printf %q "$SOURCE_DIR") && exec clangd --compile-commands-dir=$(printf %q "$CLANGD_DIR")$ARGS"
