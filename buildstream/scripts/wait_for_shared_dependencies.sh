#!/bin/bash
#
# Waits for the Linux builds of the pipeline to push the dependencies a cross
# build shares with them, rather than building them a second time, at the same
# time and on the same runner: the elements its build plan holds under the same
# cache key as the Linux one, qt6_host.bst and the other tools run on the build
# machine first. They are told apart on each run, so a new element needs no
# declaring anywhere.
#
# Usage: wait_for_shared_dependencies.sh <bst options>
#
# LINUX_DEPENDENCY_JOBS  the jobs that build the Linux dependencies and push each
#                        one as soon as it is there (see push_dependencies.sh),
#                        separated by commas.
#
# Returns once nothing shared is missing, or once none of those jobs is left to
# push what still is, which the build then makes itself. Whatever goes wrong
# here only costs that duplicate work: this script never fails.

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
# .common.sh puts bst on the PATH.
{ source .common.sh ; } 1>&2

OPTIONS=("$@")
# The options the Linux dependencies are built with, by every job of
# LINUX_DEPENDENCY_JOBS.
LINUX_OPTIONS=(--option target_triple x86_64-linux-gnu --option cxx_compiler clang++)
POLL_SECONDS=60
# qt6_host.bst, by far the largest shared element, took the Linux builds 24
# minutes in pipeline 2897906408: waiting much longer than that means something
# else holds them up, and building here then costs less.
MAX_WAIT_SECONDS=$((60 * 60))

# Prints those of the given elements that neither this cache nor the pool holds.
still_missing() {
    bst "${OPTIONS[@]}" artifact pull "$@" > /dev/null 2>&1
    # Some states take two words, such as "fetch needed".
    bst "${OPTIONS[@]}" show --deps none --format '%{name}|%{state}' "$@" \
        | awk -F '|' '$2 != "cached" { print $1 }'
}

# Whether a job of LINUX_DEPENDENCY_JOBS can still push anything. The jobs of a
# public project read without a token; an answer that does not come counts as
# no, which only costs the wait.
linux_jobs_active() {
    curl --silent --fail --max-time 30 \
         "$CI_API_V4_URL/projects/$CI_PROJECT_ID/pipelines/$CI_PIPELINE_ID/jobs?per_page=100" \
        | jq --exit-status --arg jobs "$LINUX_DEPENDENCY_JOBS" '
              [.[] | select(.name as $name | $jobs | split(",") | index($name))
                   | select(.status as $status
                            | ["created", "waiting_for_resource", "preparing", "pending",
                               "running", "scheduled"] | index($status))]
              | length > 0' > /dev/null
}

LINUX_KEYS="$(bst "${LINUX_OPTIONS[@]}" show --deps build --format '%{full-key}' squey.bst)" \
    || exit 0
mapfile -t SHARED < <(bst "${OPTIONS[@]}" show --deps build --format '%{full-key} %{name}' squey.bst \
                      | awk 'NR == FNR { linux[$1]; next } $1 in linux { print $2 }' \
                            <(printf '%s\n' "$LINUX_KEYS") -)
[ ${#SHARED[@]} -gt 0 ] || exit 0
mapfile -t MISSING < <(still_missing "${SHARED[@]}")
echo "${#SHARED[@]} dependencies shared with the Linux build, ${#MISSING[@]} of them missing."

DEADLINE=$((SECONDS + MAX_WAIT_SECONDS))
while [ ${#MISSING[@]} -gt 0 ]; do
    # Asked before looking at the pool: a job found finished pushed all it ever will.
    if ! linux_jobs_active || [ $SECONDS -ge $DEADLINE ]; then
        mapfile -t MISSING < <(still_missing "${MISSING[@]}")
        break
    fi
    echo "Waiting for the Linux builds to push: ${MISSING[*]}"
    sleep "$POLL_SECONDS"
    mapfile -t MISSING < <(still_missing "${MISSING[@]}")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Building here what the Linux builds did not push: ${MISSING[*]}"
fi
exit 0
