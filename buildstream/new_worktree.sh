#!/bin/bash

# Creates a git worktree of Squey next to the checkout it is run from, clones its
# submodules from the local copies rather than from the network, and configures
# its build tree in the Linux development sandbox.
#
# Usage: buildstream/new_worktree.sh <branch> [<base>]
#
# <branch>  branch to create, which also names the worktree directory
# <base>    commit to start from (default: origin/main, as last fetched)
#
# SQUEY_SANDBOX_SSH       ssh destination of the Linux sandbox (default: SqueyLinux)
# SQUEY_SANDBOX_SSH_OPTS  extra ssh options, e.g. "-o IdentityFile=~/.ssh/<key>"

set -e

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <branch> [<base>]" >&2
    exit 1
fi
BRANCH="$1"
BASE="${2:-origin/main}"

CHECKOUT="$(git rev-parse --show-toplevel)"
WORKTREE="$(dirname "$CHECKOUT")/${BRANCH//\//_}"
MODULES="$(git rev-parse --path-format=absolute --git-common-dir)/modules"

echo "Starting from $(git log -1 --format='%h %s' "$BASE")"
git worktree add --no-track -b "$BRANCH" "$WORKTREE" "$BASE"

# The URL overrides are not saved: 'submodule init' then registers the URLs of
# .gitmodules, and 'submodule sync' points the clones back to them.
OVERRIDES=()
while read -r NAME; do
    if [ -d "$MODULES/$NAME" ]; then
        OVERRIDES+=(-c "submodule.$NAME.url=$MODULES/$NAME")
    fi
done < <(git -C "$WORKTREE" config -f .gitmodules --name-only --get-regexp '^submodule\..*\.path$' \
             | sed 's/^submodule\.\(.*\)\.path$/\1/')
# A local copy lacks the commits recorded since it was last fetched: the update
# then stops at the first one, and leaves the submodules it had not reached yet
# cloned but not checked out.
LOCAL_COPIES_COMPLETE=true
git -C "$WORKTREE" -c protocol.file.allow=always "${OVERRIDES[@]}" submodule update --init 2> /dev/null ||
    LOCAL_COPIES_COMPLETE=false
git -C "$WORKTREE" submodule init
git -C "$WORKTREE" submodule sync
if [ "$LOCAL_COPIES_COMPLETE" = false ]; then
    # From the actual remotes this time. --force also checks out the submodules
    # whose HEAD already names the recorded commit, as an unfinished clone can.
    echo "Fetching the submodule commits the local copies lack" >&2
    git -C "$WORKTREE" submodule update --force
fi

read -r -a SSH_OPTS <<< "$SQUEY_SANDBOX_SSH_OPTS"
SSH=(ssh -T -o BatchMode=yes -o ClearAllForwardings=yes -o LogLevel=ERROR "${SSH_OPTS[@]}"
     "${SQUEY_SANDBOX_SSH:-SqueyLinux}")
if "${SSH[@]}" true < /dev/null 2> /dev/null; then
    "${SSH[@]}" "cd $(printf %q "$WORKTREE/src") && export PKG_CONFIG_PATH=\${PKG_CONFIG_PATH:-/app/lib/pkgconfig:} && cmake --preset linux-release" < /dev/null
else
    echo "The Linux sandbox is not running: configure the build tree later, with" \
         "'cmake --preset linux-release' run from $WORKTREE/src." >&2
fi

echo "$WORKTREE is ready."
