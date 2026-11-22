#!/bin/bash
#
# Builds a tree of Squey in the devcontainer, then lays out a Flatpak build
# directory out of it and of the prefix it was built against, with the scripts
# the BuildStream elements lay out theirs with. devcontainer_flatpak.sh runs it
# in a container of the image, and exports the result on the host.
#
# Usage: stage_flatpak.sh <build tree> <build directory>
#
# <build tree>       a tree .devcontainer/configure_builds.sh configured
# <build directory>  where to lay out files/; it must not exist
#
# Squey is installed, completed and stripped as squey.bst does, pruned as
# squey-cleanup.bst does, and given the desktop integration of
# org.squey.Squey_export.bst. The rest of the prefix is the part the runtime
# dependencies of Squey install when the image of the devcontainer lists it in
# RUNTIME_FILES below, and the whole prefix otherwise: build dependencies
# included, but nothing missing.

set -e
set -o pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <build tree> <build directory>" >&2
  exit 1
fi
BUILD_TREE="$(cd "$1" && pwd)"
BUILD_DIR="$2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
FLATPAK_FILES="$DIR/files/flatpak"
RUNTIME_FILES=/usr/share/squey-devcontainer/runtime-files

for variable in PREFIX TARGET_TRIPLE HOST; do
  if [ -z "${!variable}" ]; then
    echo "$variable is not set: run this in the devcontainer, as devcontainer_flatpak.sh does." >&2
    exit 1
  fi
done
if [ -e "$BUILD_DIR" ]; then
  echo "$BUILD_DIR already exists." >&2
  exit 1
fi

# A tree configured before configure_builds.sh named the library directory
# installs into lib64, where the Flatpak looks for no library, and the path of
# the plugins compiled in follows it.
if [ "$(sed -n 's/^CMAKE_INSTALL_LIBDIR:[A-Z]*=//p' "$BUILD_TREE/CMakeCache.txt")" != lib ]; then
  echo "Configuring $BUILD_TREE to install into lib, as the packages do"
  cmake -DCMAKE_INSTALL_LIBDIR=lib "$BUILD_TREE"
fi
cmake --build "$BUILD_TREE"

mkdir -p "$BUILD_DIR/files"
FILES="$(cd "$BUILD_DIR" && pwd)/files"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

if [ -f "$RUNTIME_FILES" ]; then
  tar -C "$PREFIX" --no-recursion -T "$RUNTIME_FILES" -cf - | tar -C "$FILES" -xf -
else
  echo "warning: no $RUNTIME_FILES, so the package takes the whole of $PREFIX, build dependencies included." >&2
  cp -a "$PREFIX/." "$FILES/"
fi

DESTDIR="$STAGE" cmake --install "$BUILD_TREE"
# What Squey installs, which prune_prefix.sh keeps in bin/: recorded as squey.bst
# records it.
{ cat "$BUILD_TREE/install_manifest.txt" && echo; } > "$STAGE$PREFIX/install_manifest.txt"
bash "$FLATPAK_FILES/install_linux_files.sh" "$STAGE$PREFIX" "$TARGET_TRIPLE" "$HOST"
bash "$FLATPAK_FILES/split_debug_info.sh" "$STAGE$PREFIX"
cp -a "$STAGE$PREFIX/." "$FILES/"

PYTHON_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
bash "$DIR/files/prune_prefix.sh" "$FILES" linux "$HOST" "$PYTHON_VERSION"
# The debug information goes to the debug extension, which flatpak-image.bst
# leaves out of the package.
rm -rf "$FILES/lib/debug"

(cd "$DIR" && ./generate_appstream_metadata.sh)
bash "$FLATPAK_FILES/install_desktop_integration.sh" "$FILES"
