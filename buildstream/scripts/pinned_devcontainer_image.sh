#!/bin/bash
#
# Prints the image .devcontainer/devcontainer.json pins, or the one the
# devcontainer.json given pins, - standing for the standard input. Whatever needs
# the pin reads it here.
#
# Usage: pinned_devcontainer_image.sh [<devcontainer.json>]

set -e

DEVCONTAINER_JSON="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.devcontainer/devcontainer.json}"

IMAGE="$(sed -n 's/^[[:space:]]*"image"[[:space:]]*:[[:space:]]*"\(.*\)".*$/\1/p' "$DEVCONTAINER_JSON")"
if [ -z "$IMAGE" ]; then
  echo "No image pinned in $DEVCONTAINER_JSON" >&2
  exit 1
fi
echo "$IMAGE"
