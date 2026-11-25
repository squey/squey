#!/bin/bash
#
# Packages a build of the devcontainer as a Flatpak, from the host: builds the
# tree in a container of the image .devcontainer/devcontainer.json pins and lays
# out the package there, with stage_flatpak.sh, then finishes and exports it
# here, where flatpak runs, with export_flatpak.sh. The package is made of
# whatever the checkout holds, which makes it fit for trying a change inside the
# sandbox of Flatpak, never for publishing.
#
# Usage: devcontainer_flatpak.sh [--build=<compiler>/<build type>]
#          [--repository=<directory>] [--branch=<branch>] [--install]
#
# --build       the tree of builds/x86_64-linux-gnu to package, which
#               .devcontainer/configure_builds.sh configures if need be;
#               Clang/RelWithDebInfo by default
# --repository  the repository to export to, which keeps the last package of
#               each branch alone; builds/flatpak/repo by default
# --branch      the branch of the package; devcontainer by default
# --install     install the package for the user too, from the remote
#               squey_devcontainer, which this points at the repository
#
# The container builds as the devcontainer does: same image, same path of the
# checkout, which the trees configured there name, and same compiler cache. It
# gets nothing else, neither the GPU nor the display nor the portal, which
# packaging needs none of: .devcontainer/prepare_gpu.sh, which gathers them, is
# not run, and a devcontainer running meanwhile keeps its own.

set -e

usage() {
  echo "Usage: $0 [--build=<compiler>/<build type>] [--repository=<directory>] [--branch=<branch>] [--install]" >&2
  exit 1
}

COMBINATION=Clang/RelWithDebInfo
REPOSITORY=
BRANCH=devcontainer
INSTALL=false
for argument in "$@"; do
  case "$argument" in
    --build=*) COMBINATION="${argument#*=}" ;;
    --repository=*) REPOSITORY="${argument#*=}" ;;
    --branch=*) BRANCH="${argument#*=}" ;;
    --install) INSTALL=true ;;
    *) usage ;;
  esac
done

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
CHECKOUT="$(dirname "$DIR")"
# shellcheck source=buildstream/files/flatpak/metadata.sh
source "$DIR/files/flatpak/metadata.sh"

ENGINE="$(command -v podman || command -v docker || true)"
if [ -z "$ENGINE" ]; then
  echo "Neither podman nor docker is installed: the build runs in a container." >&2
  exit 1
fi
if ! command -v flatpak > /dev/null; then
  echo "flatpak is not installed: it finishes and exports the package on the host." >&2
  exit 1
fi
# The files written in the checkout have to belong to the user, as the
# devcontainer CLI arranges with either engine.
if "$ENGINE" --version | grep -q podman; then
  USER_OPTIONS=(--userns=keep-id)
else
  USER_OPTIONS=(--user "$(id -u):$(id -g)")
fi

IMAGE="$("$DIR/scripts/pinned_devcontainer_image.sh")"
WORKSPACE="/workspaces/$(basename "$CHECKOUT")"
REPOSITORY="${REPOSITORY:-$CHECKOUT/builds/flatpak/repo}"
mkdir -p "$CHECKOUT/builds/flatpak"
STAGING="$(mktemp -d "$CHECKOUT/builds/flatpak/staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

# label=disable as the devcontainer has it, without which SELinux keeps the
# container from reading the checkout. squey-ccache is the volume of the
# compiler cache .devcontainer/devcontainer.json mounts.
# shellcheck disable=SC2016 # expanded by the shell of the container
"$ENGINE" run --rm --init "${USER_OPTIONS[@]}" --security-opt label=disable \
  --volume "$CHECKOUT:$WORKSPACE" \
  --volume squey-ccache:/home/dev/.cache/ccache \
  --env SOURCE_DIR="$WORKSPACE" --workdir "$WORKSPACE" \
  "$IMAGE" bash -ec '.devcontainer/configure_builds.sh "$1"
                     buildstream/scripts/stage_flatpak.sh "builds/$TARGET_TRIPLE/$1" "$2"' \
  - "$COMBINATION" "${STAGING#"$CHECKOUT"/}/build"

# An archive repository, the kind build-export creates when there is none,
# compresses every file of the package, those it holds already included: most
# of the time of a package whose tree has nothing to build. ostree, which
# creates an uncompressed one, does not come with flatpak everywhere.
if [ ! -e "$REPOSITORY" ] && command -v ostree > /dev/null; then
  mkdir -p "$(dirname "$REPOSITORY")"
  ostree init --repo="$REPOSITORY" --mode=bare-user-only
fi
"$DIR/scripts/export_flatpak.sh" "$STAGING/build" "$REPOSITORY" "$BRANCH"
# Uncompressed, the repository would grow by whole files at each package.
flatpak build-update-repo --prune --prune-depth=0 "$REPOSITORY" > /dev/null
echo "Exported $FLATPAK_APP_ID//$BRANCH to $REPOSITORY"

if [ "$INSTALL" = true ]; then
  REMOTE=squey_devcontainer
  URL="file://$(cd "$REPOSITORY" && pwd)"
  if flatpak --user remotes --columns=name | grep -qxF "$REMOTE"; then
    flatpak --user remote-modify --url="$URL" "$REMOTE"
  else
    flatpak --user remote-add --no-gpg-verify "$REMOTE" "$URL"
  fi
  flatpak --user install --or-update --noninteractive "$REMOTE" "$FLATPAK_APP_ID//$BRANCH"
  echo "Run it with: flatpak run $FLATPAK_APP_ID//$BRANCH"
fi
