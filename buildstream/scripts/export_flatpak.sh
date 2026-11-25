#!/bin/bash
#
# Finishes a Flatpak build directory, then exports it to a repository.
#
# Usage: export_flatpak.sh <build directory> <repository> <branch> [<GPG key>]
#
# The build directory holds the files/ of the application, as
# flatpak/org.squey.Squey.bst and stage_flatpak.sh lay them out. Whatever
# metadata it holds, this writes the one of files/flatpak/metadata.sh, which
# Flathub builds with too, and "flatpak build-finish" exports what the host reads
# out of files/share: the desktop entry, the icons, the MIME types and the
# metainfo. The commit is signed when a GPG key is given.

set -e

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <build directory> <repository> <branch> [<GPG key>]" >&2
  exit 1
fi
BUILD_DIR="$1"
REPOSITORY="$2"
BRANCH="$3"
GPG_SIGN_KEY="$4"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck source=buildstream/files/flatpak/metadata.sh
source "$DIR/files/flatpak/metadata.sh"
RUNTIME_BRANCH="$("$DIR/scripts/runtime_branch.sh")"
ARCH="$(flatpak --default-arch)"

cat > "$BUILD_DIR/metadata" <<EOF
[Application]
name=$FLATPAK_APP_ID
runtime=$FLATPAK_RUNTIME/$ARCH/$RUNTIME_BRANCH
sdk=$FLATPAK_SDK/$ARCH/$RUNTIME_BRANCH

[Extension $FLATPAK_APP_ID.Debug]
directory=lib/debug
EOF
# Where the debug extension is mounted.
mkdir -p "$BUILD_DIR/files/lib/debug"

# build-finish refuses a directory that has an export/, which it takes for one it
# already finished, and flatpak_image creates it, empty.
if [ -d "$BUILD_DIR/export" ]; then
  rmdir "$BUILD_DIR/export"
fi
flatpak build-finish --command="$FLATPAK_COMMAND" "${FLATPAK_PERMISSIONS[@]}" "$BUILD_DIR"
# build-export creates the repository, but not the directories above it.
mkdir -p "$REPOSITORY"
flatpak build-export ${GPG_SIGN_KEY:+"--gpg-sign=$GPG_SIGN_KEY"} --files=files \
  "$REPOSITORY" "$BUILD_DIR" "$BRANCH"
