#!/bin/bash

usage() {
echo "Usage: $0 [--branch=<branch_name>] [--tag=<tag_name>] [--build_type=<build_type>] [--compiler=<cxx_compiler>]"
echo "                  [--user-target=<USER_TARGET>] [--repo=<repository_path>] [--upload=<upload_url>] [--port=<scp_port>]" 1>&2; exit 1;
}

OPTS=`getopt -o r:m:b:a:t:c:u:p --long repo:,branch:,tag:,build-type:,user-target:,compiler:,upload:,port: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi

eval set -- "$OPTS"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BRANCH_NAME=master
TAG_NAME=
BUILD_TYPE=Release
USER_TARGET=developer
CXX_COMPILER=/usr/lib/sdk/gcc7/bin/g++
EXPORT_BUILD=false
REPO_DIR=
UPLOAD_URL=
UPLOAD_PORT=22

while true; do
  case "$1" in
    -b | --branch ) BRANCH_NAME="$2"; shift 2 ;;
    -a | --tag ) TAG_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -m | --user-target ) USER_TARGET="$2"; shift 2 ;;
    -c | --compiler ) CXX_COMPILER="$2"; shift 2 ;;
    -r | --repo ) EXPORT_BUILD=true; REPO_DIR="$2"; shift 2 ;;
    -u | --upload ) UPLOAD_URL="$2"; shift 2 ;;
    -p | --port ) UPLOAD_PORT="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

echo "BRANCH_NAME"=$BRANCH_NAME
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ $CURRENT_BRANCH == "HEAD" ]; then
    git fetch -a
    git checkout -B $BRANCH_NAME origin/$BRANCH_NAME
else
    BRANCH_NAME=$CURRENT_BRANCH
fi

# Manualy substitute variables since flatpak-builder doesn't seem to support this yet as version 0.10.8
if [ ! -z "$TAG_NAME" ]; then
    sed -e "s/@@BUILD_TYPE@@/$BUILD_TYPE/g" -e "s/@@USER_TARGET@@/$USER_TARGET/g" -e "s#@@CXX_COMPILER@@#$CXX_COMPILER#g" -e "/@@BRANCH_NAME@@/c\          \"tag\": \"$TAG_NAME\"" $DIR/inendi-inspector.json.in > $DIR/inendi-inspector.json || exit 2
else
    sed -e "s/@@BUILD_TYPE@@/$BUILD_TYPE/g" -e "s/@@USER_TARGET@@/$USER_TARGET/g" -e "s#@@CXX_COMPILER@@#$CXX_COMPILER#g" -e "s/@@BRANCH_NAME@@/$BRANCH_NAME/g" -e "s/\"branch\": \"master\"/\"branch\": \"$BRANCH_NAME\"/g" $DIR/inendi-inspector.json.in > $DIR/inendi-inspector.json || exit 3
fi
