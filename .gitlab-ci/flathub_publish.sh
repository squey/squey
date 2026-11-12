#!/bin/bash
#
# Publishes the release VERSION.txt names on Flathub.
#
# Flathub builds the application from the manifest of the flathub/org.squey.Squey
# repository on GitHub, whose master branch only moves through pull requests that
# passed its test build. This pushes a branch whose manifest points at the archive
# the "pages" job put online, opens its pull request, waits for the statuses the
# protection of master requires, then merges it. Flathub publishes the release an
# hour or two later, unless it holds it for review.
#
# A run after a failed one takes over its branch and pull request, without
# restarting the build of a manifest that did not change.
#
# FLATHUB_GITHUB_PR_TOKEN       a personal access token (classic) of a maintainer
#                               of the repository, with the "public_repo" scope.
#                               A fine-grained token cannot write to the repository
#                               of an organization its owner is only an outside
#                               collaborator of.
# FLATHUBGITCREDENTIALUSERNAME  the name and email the commit is authored with.
# FLATHUBGITCREDENTIALEMAIL
# FLATHUB_BUILD_TIMEOUT         how many minutes to wait for the test build, 150 by
#                               default.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

repository=flathub/org.squey.Squey
version=$(tr -d '\n' < VERSION.txt)
branch="publish-$version"
archive_url="https://squey.gitlab.io/squey/squey_flatpak_$version.tar.xz"
build_timeout=${FLATHUB_BUILD_TIMEOUT:-150}

if [ -z "${FLATHUB_GITHUB_PR_TOKEN:-}" ]; then
    echo "FLATHUB_GITHUB_PR_TOKEN is not set: a protected variable only reaches the pipelines of protected branches and tags" >&2
    exit 1
fi
# gh uses a token it finds there as it is, while "gh auth login" would refuse one
# without the "repo" and "read:org" scopes, which reach every private repository
# of the account.
export GH_TOKEN="$FLATHUB_GITHUB_PR_TOKEN"
export GIT_TERMINAL_PROMPT=0

if [ "$(gh api "repos/$repository" --jq .permissions.push)" != true ]; then
    echo "The owner of FLATHUB_GITHUB_PR_TOKEN cannot push to $repository" >&2
    exit 1
fi

# The checksum of the file the Flathub builders download, which the deployment of
# the "pages" job serves a few minutes after the job ends.
sha256=
for attempt in $(seq 30); do
    if sha256=$(curl --fail --silent --show-error --location "$archive_url" | sha256sum); then
        sha256=${sha256%% *}
        break
    fi
    sha256=
    echo "$archive_url is not online yet ($attempt/30)"
    sleep 60
done
if [ -z "$sha256" ]; then
    echo "$archive_url did not come online" >&2
    exit 1
fi
echo "$archive_url: sha256 $sha256"

checkout=$(mktemp -d)
git clone --quiet "https://github.com/$repository.git" "$checkout"
if grep -qF "$archive_url" "$checkout/org.squey.Squey.yaml" && grep -qF "$sha256" "$checkout/org.squey.Squey.yaml"; then
    echo "Flathub already builds $version"
    exit 0
fi
cp buildstream/flathub/flathub.json "$checkout"
jinja2 -D sha256="$sha256" -D version="$version" buildstream/flathub/org.squey.Squey.yaml.j2 > "$checkout/org.squey.Squey.yaml"

cd "$checkout"
git checkout --quiet -b "$branch"
git -c user.name="$FLATHUBGITCREDENTIALUSERNAME" -c user.email="$FLATHUBGITCREDENTIALEMAIL" \
    commit --quiet --all --message "Publish version $version"
git --no-pager show --stat --patch HEAD

if git fetch --quiet origin "refs/heads/$branch" 2> /dev/null && git diff --quiet FETCH_HEAD HEAD; then
    head=$(git rev-parse FETCH_HEAD)
    echo "$branch already holds this manifest"
else
    # The helper reads the token from the environment, out of the command line and of
    # any file.
    # shellcheck disable=SC2016
    git -c credential.helper='!f() { test "$1" = get && echo username=x-access-token && echo "password=$GH_TOKEN"; }; f' \
        push --force origin "$branch"
    head=$(git rev-parse HEAD)
fi

pr=$(gh pr list --repo "$repository" --head "$branch" --state open --json number --jq '.[0].number // empty')
if [ -z "$pr" ]; then
    gh pr create --repo "$repository" --base master --head "$branch" --title "Publish version $version" \
        --body "Squey $version, from $archive_url${CI_JOB_URL:+ (opened by $CI_JOB_URL)}"
    pr=$(gh pr list --repo "$repository" --head "$branch" --state open --json number --jq '.[0].number')
fi
pr_url="https://github.com/$repository/pull/$pr"

mapfile -t required < <(gh api "repos/$repository/branches/master" --jq '.protection.required_status_checks.contexts[]')
if [ "${#required[@]}" -eq 0 ]; then
    echo "Could not read the statuses the protection of master requires" >&2
    exit 1
fi
echo "Waiting for ${required[*]} on $pr_url"
deadline=$(($(date +%s) + build_timeout * 60))
reported=none
while :; do
    statuses=$(gh api "repos/$repository/commits/$head/status" --jq '.statuses[] | "\(.context) \(.state) \(.description // "")"')
    if [ "$statuses" != "$reported" ]; then
        echo "$(date -u +%H:%M:%S) ${statuses:-no status yet}" | sed '2,$s/^/         /'
        reported=$statuses
    fi
    pending=false
    for context in "${required[@]}"; do
        state=$(awk -v context="$context" '$1 == context { print $2 }' <<< "$statuses")
        case "$state" in
            success) ;;
            failure | error)
                echo "$context is $state, see $pr_url" >&2
                exit 1
                ;;
            *) pending=true ;;
        esac
    done
    [ "$pending" = true ] || break
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "Flathub did not build $pr_url within $build_timeout minutes: merge it once built, or run this job again, which waits for the same build" >&2
        exit 1
    fi
    sleep 60
done

gh pr merge "$pr" --repo "$repository" --merge
echo "Merged $pr_url: Flathub publishes $version once its official build is done"
