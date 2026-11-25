#!/bin/bash
#
# Prints the branch of the freedesktop runtime Squey runs on, as the
# "runtime-branch" variable of buildstream/app.yml gives it to the BuildStream
# elements. Whatever names that branch outside of BuildStream reads it here.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

BRANCH="$(sed -n "s/^[[:space:]]*runtime-branch:[[:space:]]*[\"']\{0,1\}\([^\"']*\)[\"']\{0,1\}[[:space:]]*$/\1/p" "$DIR/app.yml")"
if [ -z "$BRANCH" ]; then
  echo "No runtime-branch in $DIR/app.yml" >&2
  exit 1
fi
echo "$BRANCH"
