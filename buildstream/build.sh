#!/bin/bash

set -e
set -x

usage() {
echo "Usage: $0 [--branch=<branch_name_or_tag_name>] [--disable-testsuite] [--user-target=<USER_TARGET>]"
echo "                  [--workspace-prefix=<prefix>] [--repo=<repository_path>] [--gpg-private-key-path=<key>]"
echo "                  [--gpg-sign-key=<key>] [--upload=<upload_url>] [--port=<scp_port>]" 1>&2; exit 1;
}

# Set default options
BRANCH_NAME=main
BRANCH_SPECIFIED=false
TAG_NAME=
BUILD_TYPE=RelWithDebInfo
USER_TARGET=developer
USER_TARGET_SPECIFIED=false
WORKSPACE_PREFIX=
EXPORT_BUILD=false
REPO_DIR="repo"
UPLOAD_URL=
UPLOAD_PORT=22
RUN_TESTSUITE=true
GPG_PRIVATE_KEY_PATH=
GPG_SIGN_KEY=

# Override default options with user provided options
OPTS=`getopt -o h:r:m:b:t:c:d:g:k:w:e --long help,flatpak-export:,flatpak-repo:,workspace-prefix:,crash-reporter-token:,gpg-private-key-path:,gpg-sign-key:,branch:,build-type:,user-target:,disable-testsuite -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -b | --branch ) BRANCH_SPECIFIED=true; BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET_SPECIFIED=true; USER_TARGET="$2"; shift 2 ;;
    -d | --disable-testsuite ) RUN_TESTSUITE=false; shift 1 ;;
    -w | --workspace-prefix ) WORKSPACE_PREFIX="$2"; shift 2 ;;
    -e | --flatpak-export ) EXPORT_BUILD="$2"; shift 2 ;;
    -r | --flatpak-repo ) REPO_DIR="$2"; shift 2 ;;
    -c | --crash-reporter-token) INSPECTOR_CRASH_REPORTER_TOKEN="$2"; shift 2 ;;
    -g | --gpg-private-key-path ) GPG_PRIVATE_KEY_PATH="$2"; shift 2 ;;
    -k | --gpg-sign-key ) GPG_SIGN_KEY="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

source .common.sh

WORKSPACE_NAME="workspace_build"
open_workspace "$WORKSPACE_NAME"

# Use proper branch if specified
ORIGIN="origin"
if git rev-parse "tags/$BRANCH_NAME"; then
    ORIGIN="tags"
fi
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ $CURRENT_BRANCH == "HEAD" -o $BRANCH_SPECIFIED = true ]; then
    pushd .
    cd "$WORKSPACE_PREFIX/$WORKSPACE_NAME"
    git reset --hard HEAD # Clean env
    git fetch -a --tags --force
    git checkout -B $BRANCH_NAME $ORIGIN/$BRANCH_NAME
    git submodule update --recursive
    popd
else
    BRANCH_NAME=$CURRENT_BRANCH
fi
if [ $USER_TARGET == "customer" ]; then
    BRANCH_NAME="main"
fi

# Fill-in crash reporter token
INSPECTOR_CRASH_REPORTER_TOKEN_FILE="$WORKSPACE_PREFIX/$WORKSPACE_NAME/libpvkernel/include/pvkernel/core/PVCrashReporterToken.h"
sed -e "s|\(INSPECTOR_CRASH_REPORTER_TOKEN\) \"\"|\1 \"$INSPECTOR_CRASH_REPORTER_TOKEN\"|" -i "$INSPECTOR_CRASH_REPORTER_TOKEN_FILE"

# Fill-in release and date
jinja2 -D version="$(cat ../VERSION.txt | tr -d '\n')" -D date="$(date --iso)" files/com.gitlab.inendi.Inspector.metainfo.xml.j2 > files/com.gitlab.inendi.Inspector.metainfo.xml

# Build INENDI Inspector
BUILD_OPTIONS=""
if [ $USER_TARGET_SPECIFIED = true ]; then
  BUILD_OPTIONS="--option user_target $USER_TARGET"
fi
if  [ "$RUN_TESTSUITE" = false ]; then
  BUILD_OPTIONS="$BUILD_OPTIONS --option disable_testsuite True"
fi
bst $BUILD_OPTIONS build inendi-inspector.bst

# Run testsuite with "bst shell" to have network access (bst hasn't a "test-commands" (yet?) like in flatpak-builder)
if  [ "$RUN_TESTSUITE" = true ]; then
    bash -c "bst $BUILD_OPTIONS shell $MOUNT_OPTS inendi-inspector.bst -- bash -c  'cp -r /build . && ln -s /tests . && cd build && run_cmd.sh ctest --output-on-failure -T test -R INSPECTOR_TEST'"
fi

# Export flatpak images
if [ "$EXPORT_BUILD" = true ]; then
  if [[ ! -z "$GPG_PRIVATE_KEY_PATH" ]]; then
    # Import GPG private key
    gpg --import --no-tty --batch --yes $GPG_PRIVATE_KEY_PATH
  fi

  # Export flatpak Release image
  rm -rf $DIR/build
  bst $BUILD_OPTIONS build flatpak/com.gitlab.inendi.Inspector.bst
  bst $BUILD_OPTIONS checkout flatpak/com.gitlab.inendi.Inspector.bst "$DIR/build"
  if [[ ! -z "$GPG_SIGN_KEY" ]]; then
    flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  else
    flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  fi

  ## Export flatpak Debug image
  #rm -rf $DIR/build
  #bst $BUILD_OPTIONS build flatpak/com.gitlab.inendi.Inspector.Debug.bst
  #bst $BUILD_OPTIONS checkout flatpak/com.gitlab.inendi.Inspector.Debug.bst "$DIR/build"
  #if [[ ! -z "$GPG_SIGN_KEY" ]]; then
  #  flatpak build-export --gpg-sign=$GPG_SIGN_KEY --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #else
  #  flatpak build-export --files=files $REPO_DIR $DIR/build $BRANCH_NAME
  #fi
fi

# Push artifacts
bst --option push_artifacts True push `ls elements -p -I "base.bst" -I "freedesktop-sdk.bst" -I "inendi-inspector*.bst" |grep -v / | tr '\n' ' '` || true

function cleanup {
  rm -rf $HOME/.cache/buildstream/artifacts/extract/inendi-inspector/inendi-inspector
  rm -rf $HOME/.cache/buildstream/build
  rm -rf /srv/tmp-inspector/tomjon/*
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT
