#!/bin/bash
#
# Rewrites, or verifies, the image reference pinned in
# .devcontainer/devcontainer.json.
#
# The pin travels with the branch, exactly like the BASE_IMAGE_NAME of
# .gitlab-ci.yml: a branch that changes the dependency graph carries the tag of
# its own image, every other branch inherits the one it forked from. That is
# what keeps the number of published images down to the number of distinct
# dependency graphs rather than to the number of branches.
#
# Usage:
#   update_devcontainer_pin.sh <image_reference>   pin that reference
#   update_devcontainer_pin.sh --verify-image      fail if the pin names an
#                                                  image the registry does not
#                                                  hold

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
REPO_DIR="$( cd "$DIR/.." && pwd )"
# Every configuration under .devcontainer/ pins the same image; the first one
# is the reference the checks read, the others are kept in step.
DEVCONTAINER_JSON="$REPO_DIR/.devcontainer/devcontainer.json"
mapfile -t DEVCONTAINER_JSONS < <(find "$REPO_DIR/.devcontainer" -name devcontainer.json | sort)

pinned_image() {
  sed -n 's/^[[:space:]]*"image"[[:space:]]*:[[:space:]]*"\(.*\)".*$/\1/p' "$DEVCONTAINER_JSON"
}

# A pin agreeing with the dependency graph says nothing about the image being
# published: without this, a merge request can go green on a pin that names
# something nobody can pull, and the devcontainer of the default branch stays
# broken until someone notices.
if [ "$1" = "--verify-image" ]; then
  pinned="$(pinned_image)"
  if skopeo inspect "docker://$pinned" &> /dev/null; then
    echo "$pinned is published."
    exit 0
  fi
  echo >&2 "devcontainer.json pins '$pinned', which the registry does not hold."
  echo >&2 "Publish it with:"
  echo >&2 "  buildstream/scripts/build_devcontainer_image.sh --push=true"
  exit 1
fi

IMAGE="$1"
[ -n "$IMAGE" ] || { echo >&2 "Usage: $0 <image_reference> | --verify-image"; exit 1; }

# Only the "image" line is touched, so that whatever else the file grows over
# time survives the rewrite untouched.
for json in "${DEVCONTAINER_JSONS[@]}"; do
  sed -i "s|^\([[:space:]]*\"image\"[[:space:]]*:[[:space:]]*\"\).*\(\"\)|\1$IMAGE\2|" "$json"
  echo "Pinned $IMAGE in ${json#$REPO_DIR/}"
done
