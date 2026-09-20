#!/bin/bash
#
# Keeps the devcontainer pin of the default branch in step with its dependency
# graph.
#
# Runs in the "ensure devcontainer image" job, once
# build_devcontainer_image.sh --update-pin=true has published the image of the
# graph and written its tag into .devcontainer/. That tag is a digest of the
# graph, so the change holds nothing to review, and finding it out takes
# BuildStream, which a contributor may not even have:
# - in a merge request, a pin that moved is only reported, the merge takes care
#   of it;
# - on the default branch, the pipeline a merge that moves a dependency starts
#   commits the pin there, pushing with a deploy key allowed to.
#
# DEVCONTAINER_PIN_DEPLOY_KEY_B64  private key of that deploy key, base64-encoded.
#                                  A protected variable: the pipelines of merge
#                                  requests never receive it.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

pinned_image() {
    sed -n 's/^[[:space:]]*"image"[[:space:]]*:[[:space:]]*"\(.*\)".*$/\1/p'
}
PINNED="$(git show HEAD:.devcontainer/devcontainer.json | pinned_image)"
TAG="$(pinned_image < .devcontainer/devcontainer.json)"
TAG="${TAG##*:}"
# Only the tag counts: in a fork, the pipeline writes the registry of the fork,
# which the pin of the branch has no reason to name.
git checkout -- .devcontainer
if [ "${PINNED##*:}" = "$TAG" ]; then
    echo "The devcontainer pin matches the dependency graph."
    exit 0
fi
IMAGE="${PINNED%:*}:$TAG"

if [ "$CI_COMMIT_BRANCH" != "$CI_DEFAULT_BRANCH" ]; then
    echo "The dependency graph resolves to $IMAGE, which the pin does not name yet:" \
         "merging into $CI_DEFAULT_BRANCH pins it there."
    exit 0
fi
if [ -z "$DEVCONTAINER_PIN_DEPLOY_KEY_B64" ]; then
    echo >&2 "$CI_DEFAULT_BRANCH resolves to $IMAGE, which its pin does not name, and this" \
             "pipeline has no DEVCONTAINER_PIN_DEPLOY_KEY_B64 to commit it with."
    exit 1
fi

buildstream/scripts/update_devcontainer_pin.sh "$IMAGE" > /dev/null
git add .devcontainer
COMMITTER=(-c user.name="Squey CI" -c user.email="noreply@squey.org")
git "${COMMITTER[@]}" commit --quiet --file=- << EOF
Pin the image $CI_DEFAULT_BRANCH resolves to

The dependency graph of $CI_DEFAULT_BRANCH resolves to $TAG, an image
the "ensure devcontainer image" job has published.
EOF

KEY_DIR="$(mktemp -d)"
trap 'rm -rf "$KEY_DIR"' EXIT
base64 -d <<< "$DEVCONTAINER_PIN_DEPLOY_KEY_B64" > "$KEY_DIR/key"
chmod 600 "$KEY_DIR/key"
# The host key GitLab publishes for gitlab.com, whose fingerprint is
# SHA256:eUXGGm1YGsMAS7vkcx6JOJdOGHPem5gQp4taiCfCLB8, so that ssh only talks to
# the server holding it. An impostor could not steal the deploy key, which ssh
# never sends, but it could hand this job a forged branch and swallow its push.
echo "gitlab.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAfuCHKVTjquxvt6CM6tdG4SLp1Btn/nOeHHE5UOzRdf" \
    > "$KEY_DIR/known_hosts"
export GIT_SSH_COMMAND="ssh -i $KEY_DIR/key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KEY_DIR/known_hosts"
REMOTE="git@gitlab.com:$CI_PROJECT_PATH.git"

for attempt in 1 2 3; do
    if git push --quiet "$REMOTE" "HEAD:refs/heads/$CI_DEFAULT_BRANCH"; then
        echo "Pinned $IMAGE on $CI_DEFAULT_BRANCH."
        exit 0
    fi
    # The branch moved on meanwhile. A merge that moved a dependency again starts
    # a pipeline of its own, which pins its own graph; any other leaves this one.
    git fetch --quiet "$REMOTE" "$CI_DEFAULT_BRANCH"
    if ! git diff --quiet HEAD~1 FETCH_HEAD -- buildstream project.conf; then
        echo "$CI_DEFAULT_BRANCH has moved a dependency since, and the pipeline of that merge pins it."
        exit 0
    fi
    if ! git "${COMMITTER[@]}" rebase --quiet FETCH_HEAD; then
        git rebase --abort
        echo >&2 "$CI_DEFAULT_BRANCH changed its pin meanwhile to something else than $IMAGE."
        exit 1
    fi
done
echo >&2 "Could not push the pin to $CI_DEFAULT_BRANCH."
exit 1
