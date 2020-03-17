#!/bin/bash

set -e
set -x

usage() {
echo "Usage: $0 [--branch=<branch_name_or_tag_name>] [--disable-testsuite] [--user-target=<USER_TARGET>]"
echo "                  [--repo=<repository_path>] [--upload=<upload_url>] [--port=<scp_port>] [--pcap-inspector]" 1>&2; exit 1;
}

source .common.sh

# Set default options
BRANCH_NAME=master
BRANCH_SPECIFIED=false
TAG_NAME=
BUILD_TYPE=RelWithDebInfo
USER_TARGET=developer
USER_TARGET_SPECIFIED=false
EXPORT_BUILD=false
EXPORT_PCAP_BUILD=false
REPO_DIR="repo"
UPLOAD_URL=
UPLOAD_PORT=22
RUN_TESTSUITE=true

# Override default options with user provided options
OPTS=`getopt -o h:r:m:b:t:c:u:d:p:i --long help,repo:,branch:,build-type:,user-target:,disable-testsuite,upload:,port,pcap-inspector -n 'parse-options' -- "$@"`
if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -b | --branch ) BRANCH_SPECIFIED=true; BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET_SPECIFIED=true; USER_TARGET="$2"; shift 2 ;;
    -d | --disable-testsuite ) RUN_TESTSUITE=false; shift 1 ;;
    -r | --repo ) EXPORT_BUILD=true; REPO_DIR="$2"; shift 2 ;;
    -u | --upload ) UPLOAD_URL="$2"; shift 2 ;;
    -p | --port ) UPLOAD_PORT="$2"; shift 2 ;;
    -i | --pcap-inspector ) EXPORT_PCAP_BUILD=true; shift 1 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

WORKSPACE_NAME="workspace_build"
open_workspace "$WORKSPACE_NAME"

# Use proper branch if specified
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ $CURRENT_BRANCH == "HEAD" -o $BRANCH_SPECIFIED = true ]; then
    pushd .
    cd "$DIR/$WORKSPACE_NAME"
    git fetch -a
    git checkout -B $BRANCH_NAME origin/$BRANCH_NAME
    git submodule update --recursive
    popd
else
    BRANCH_NAME=$CURRENT_BRANCH
fi
if [ $USER_TARGET == "customer" ]; then
    BRANCH_NAME="master"
fi

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
    bash -c "bst shell $MOUNT_OPTS inendi-inspector.bst -- bash -c  'cp -r /build . && ln -s /tests . && cd build && run_cmd.sh ctest --output-on-failure -T test -R INSPECTOR_TEST'"
fi

# Export flatpak image
rm -rf $DIR/build
bst $BUILD_OPTIONS build flatpak/com.esi_inendi.Inspector.bst
bst checkout flatpak/com.esi_inendi.Inspector.bst "$DIR/build"
flatpak build-export --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir="$DIR/gnupg" --files=files $REPO_DIR $DIR/build $BRANCH_NAME


# Export flatpak Debug image
rm -rf $DIR/build
bst $BUILD_OPTIONS build flatpak/com.esi_inendi.Inspector.Debug.bst
bst checkout flatpak/com.esi_inendi.Inspector.Debug.bst "$DIR/build"
flatpak build-export --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir="$DIR/gnupg" --files=files $REPO_DIR $DIR/build $BRANCH_NAME


if [ $EXPORT_PCAP_BUILD = true ]; then
    # Export flatpak image
    rm -rf $DIR/build
    bst $BUILD_OPTIONS build flatpak/com.pcap_inspector.Inspector.bst
    bst checkout flatpak/com.pcap_inspector.Inspector.bst "$DIR/build"
    flatpak build-export --gpg-sign=3C88A1109C7272D88C1DA28ABEEF7E7DF6D0F465 --gpg-homedir="$DIR/gnupg" --files=files $REPO_DIR $DIR/build $BRANCH_NAME
fi

# Upload flatpak image to remote repository
if [ ! -z "$UPLOAD_URL" -a ! -z "$REPO_DIR" ]; then
    $DIR/scripts/ostree-releng-scripts/rsync-repos --rsync-opt "-e ssh -p $UPLOAD_PORT" --src $REPO_DIR/ --dest $UPLOAD_URL
fi

# Push artifacts
bst --option push_artifacts True push `ls elements -p -I "base.bst" -I "freedesktop-sdk.bst" -I "inendi-inspector*.bst" |grep -v / | tr '\n' ' '`

function cleanup {
  rm -rf $XDG_CONFIG_HOME/buildstream/artifacts/extract/inendi-inspector/inendi-inspector
  rm -rf $XDG_CONFIG_HOME/buildstream/build
  rm -rf /srv/tmp-inspector/tomjon/*
}

trap cleanup EXIT SIGKILL SIGQUIT SIGSEGV SIGABRT
