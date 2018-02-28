#!/bin/bash

usage() { echo "Usage: $0 [--branch=<branch_name>] [--build_type=<build_type>] [--compiler=<cxx_compiler>]" 1>&2; exit 1; }

OPTS=`getopt -o b:t:c: --long branch:,build-type:,compiler: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then usage >&2 ; exit 1 ; fi

eval set -- "$OPTS"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BRANCH_NAME=master
BUILD_TYPE=Release
CXX_COMPILER=g++

while true; do
  case "$1" in
    -b | --branch ) BRANCH_NAME="$2"; shift 2 ;;
    -t | --build-type ) BUILD_TYPE="$2"; shift 2 ;;
    -c | --compiler ) CXX_COMPILER="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ $CURRENT_BRANCH == "HEAD" ]; then
    git fetch -a
    git checkout -B $BRANCH_NAME origin/$BRANCH_NAME
else
    BRANCH_NAME=$CURRENT_BRANCH
fi

# Manualy substitute variables since flatpak-builder doesn't seem to support this yet as version 0.10.8
sed -e "s/@@BUILD_TYPE@@/$BUILD_TYPE/g" -e "s/@@CXX_COMPILER@@/$CXX_COMPILER/g" -e "s/@@BRANCH_NAME@@/$BRANCH_NAME/g" $DIR/inendi-inspector.json.in > $DIR/inendi-inspector.json || exit 1
