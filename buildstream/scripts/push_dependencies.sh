#!/bin/bash
#
# Builds the dependencies of squey.bst, handing each one over to the shared
# artifact cache (ARTIFACT_CACHE_URL, the pool the CI runner hosts) as soon as it
# is there, where later builds find it instead of building it again. The cross
# builds of a pipeline can be waiting for those they share with the Linux one,
# see wait_for_shared_dependencies.sh.
#
# Usage: push_dependencies.sh <bst options>
#
# The remotes of the pool are declared pull-only in the runner configuration, so
# a build never uploads anything by itself: this is the only place that pushes,
# and it leaves squey.bst out. That one is the largest artifact of the build and
# the next commit makes it obsolete, whereas its dependencies are worth the room
# they take, the freedesktop-sdk included: once upstream retention drops them,
# the alternative is to build them from source again. Sources stay out of the
# pool altogether, they are quick to fetch and stable enough to not be worth the
# room.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# .common.sh puts bst on the PATH. It reads $? right after a grep to validate
# TARGET_TRIPLE, which the errexit of this script would preempt; lift it for the
# duration of the source.
set +e
{ source .common.sh ; } 1>&2
set -e

# "--deps build" is the whole build plan of squey.bst, minus the element itself
DEPENDENCIES=$(bst "$@" show --deps build --format '%{name}' squey.bst)
# A push remote makes the build push every element of the plan once cached,
# whether built or pulled. A pool that cannot be reached is left out with a
# warning, but a push failing on the way stops the build: it is then finished
# without the pool, as a failed push costs a redundant rebuild later on, never a
# job. A build failure of its own fails the second run as well, from the cache.
# shellcheck disable=SC2086 # one element name per word
bst "$@" build --retry-failed --artifact-remote "$ARTIFACT_CACHE_URL" $DEPENDENCIES ||
    bst "$@" build $DEPENDENCIES
