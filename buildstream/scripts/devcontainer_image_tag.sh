#!/bin/bash
#
# Prints the tag of the devcontainer image matching the current dependency graph.
#
# BuildStream already computes a cache key for every element, so the tag is a
# digest of the keys of all the build dependencies of squey.bst. "--deps build"
# is the build plan minus squey.bst itself, which makes the tag invariant to
# anything under src/: a commit that only touches the sources resolves to the
# very same image, and only a change to the dependency graph calls for
# publishing a new one. The script that builds the image is folded into the
# digest too, otherwise changing the image layout without touching a dependency
# would leave the published image stale under an unchanged tag.

# pipefail matters here: the digest is computed through a pipeline, whose exit
# status is that of sha256sum alone. Without it a failing "bst show" yields the
# perfectly stable digest of an empty dependency list, and the tag silently
# stops describing anything.
set -e
set -o pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

# The devcontainer only ever covers a native Linux build. Cross-compilation and
# packaging stay in BuildStream, so the other target triples never get an image.
TARGET_TRIPLE=x86_64-linux-gnu
CXX_COMPILER=clang++
export TARGET_TRIPLE

# .common.sh activates the python virtual environment that holds bst, and
# reports on the GPU drivers it finds on the way: keep whatever it prints away
# from stdout, which carries the tag and nothing else.
# .common.sh reads $? right after a grep to validate TARGET_TRIPLE, which the
# errexit of this script would preempt; lift it for the duration of the source.
set +e
{ source .common.sh ; } 1>&2
set -e

# Sorted so that reordering the "depends" list of squey.bst, which changes the
# topological order without changing the closure, keeps resolving to the same
# image. LC_ALL=C is what makes that sort reproducible: element names are full
# of '-', '_', '.' and '/', which a UTF-8 collation and the C one order
# differently, so a developer on a en_US.UTF-8 host and the CI job on a C one
# would otherwise derive two different tags from the very same graph.
DEPENDENCIES="$(bst --option target_triple "$TARGET_TRIPLE" \
                    --option cxx_compiler "$CXX_COMPILER" \
                    show --deps build --format '%{name}|%{full-key}' squey.bst \
                | LC_ALL=C sort)"

# A graph this small is a broken checkout rather than a real one, and hashing it
# would pin the image of a project that does not exist.
if [ "$(printf '%s\n' "$DEPENDENCIES" | wc -l)" -lt 100 ]; then
  echo >&2 "bst reported an implausibly small build closure for squey.bst; refusing to derive a tag from it."
  exit 1
fi

{
  printf '%s\n' "$DEPENDENCIES"
  # Only what decides the content of the image belongs in the digest. The
  # scripts that merely run inside the container, configure_builds.sh among
  # them, are read from the workspace at run time: folding them in here would
  # republish 12 GB over a comment.
  cat scripts/build_devcontainer_image.sh
} | sha256sum | cut -c1-16
