#!/bin/bash
#
# Publishes the release VERSION.txt names on GitLab or on GitHub, as the releases
# published by hand on GitHub so far: named "Squey <version>", with the entry of
# the version in CHANGELOG for notes. A release published already gets that name
# and those notes again.
#
# Usage: publish_release.sh gitlab|github
#
# The release goes to the tag release-<version>, which must exist: this never
# creates one, as pushing such a tag starts the production of a release. GitHub
# receives the tags from the push mirror of the GitLab repository.
#
# CI_JOB_TOKEN          the token of the job, for GitLab: the user who started the
#                       pipeline needs the Developer role at least.
# GITHUB_RELEASE_TOKEN  for GitHub, a fine-grained personal access token limited to
#                       squey/squey, with the "Contents" permission in read and
#                       write.

set -euo pipefail

target=${1:-}
if [ "$target" != gitlab ] && [ "$target" != github ]; then
    echo "Usage: $0 gitlab|github" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

version=$(tr -d '\n' < VERSION.txt)
tag="release-$version"
name="Squey $version"

# The entry of a version ending in .0 is headed by its first two numbers alone, as
# in "5.2 (2026-10-08)", and its heading is underlined with dashes.
notes=$(mktemp)
awk -v long="$version (" -v short="${version%.0} (" '
    !found && (index($0, long) == 1 || index($0, short) == 1) { found = 1; getline; next }
    found && /^[0-9]+\.[0-9]+(\.[0-9]+)? \(/ { exit }
    found { lines[++n] = $0 }
    END {
        first = 1
        while (first <= n && lines[first] ~ /^[ \t\r]*$/) first++
        last = n
        while (last >= first && lines[last] ~ /^[ \t\r]*$/) last--
        for (i = first; i <= last; i++) print lines[i]
    }' CHANGELOG > "$notes"
if [ ! -s "$notes" ]; then
    echo "CHANGELOG has no entry for $version" >&2
    exit 1
fi

case "$target" in
gitlab)
    api="$CI_API_V4_URL/projects/$CI_PROJECT_ID"
    auth=(--header "JOB-TOKEN: $CI_JOB_TOKEN")
    if ! curl --fail --silent "${auth[@]}" "$api/repository/tags/$tag" > /dev/null; then
        echo "The GitLab repository has no tag $tag" >&2
        exit 1
    fi
    release=$(jq --null-input --arg tag "$tag" --arg name "$name" --rawfile description "$notes" \
        '{tag_name: $tag, name: $name, description: $description}')
    if curl --fail --silent "${auth[@]}" "$api/releases/$tag" > /dev/null; then
        request=(--request PUT "$api/releases/$tag")
    else
        request=(--request POST "$api/releases")
    fi
    curl --fail-with-body --silent --show-error "${auth[@]}" --header "Content-Type: application/json" \
        --data "$release" "${request[@]}" > /dev/null
    echo "Published $CI_PROJECT_URL/-/releases/$tag"
    ;;
github)
    if [ -z "${GITHUB_RELEASE_TOKEN:-}" ]; then
        echo "GITHUB_RELEASE_TOKEN is not set: a protected variable only reaches the pipelines of protected branches and tags" >&2
        exit 1
    fi
    export GH_TOKEN="$GITHUB_RELEASE_TOKEN"
    repository=squey/squey
    # The mirror pushes a tag a few minutes after it reaches GitLab.
    for attempt in $(seq 10); do
        if gh api "repos/$repository/git/ref/tags/$tag" --silent 2> /dev/null; then
            break
        elif [ "$attempt" -eq 10 ]; then
            echo "$repository has no tag $tag: check the push mirror of the GitLab repository" >&2
            exit 1
        fi
        echo "$repository has no tag $tag yet ($attempt/10)"
        sleep 60
    done
    if gh release view "$tag" --repo "$repository" > /dev/null 2>&1; then
        gh release edit "$tag" --repo "$repository" --title "$name" --notes-file "$notes" > /dev/null
    else
        gh release create "$tag" --repo "$repository" --verify-tag --title "$name" --notes-file "$notes" > /dev/null
    fi
    echo "Published https://github.com/$repository/releases/tag/$tag"
    ;;
esac
