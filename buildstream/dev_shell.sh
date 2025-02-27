#!/bin/bash

usage() {
    echo "Usage: $0 "
    echo "--help                          : show help"
    echo "--target_triple=<TARGET_TRIPLE> : cross-compile for the given target triple" 1>&2; exit 1;
}

OPTS=`getopt -o h,t --long help,target_triple: -n 'parse-options' -- "$@"`
if [ $? -ne 0 ] ; then usage >&2 ; exit 1 ; fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -h | --help ) usage >&2 ; exit 0 ;;
    -t | --target_triple ) TARGET_TRIPLE="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

source .common.sh

scripts/cuda-devices &> /dev/null || true

[ -n "$TARGET_TRIPLE" ] && TARGET_OPTION="--option target_triple $TARGET_TRIPLE"

bst $TARGET_OPTION build base.bst
if [ command -v "waypipe" &> /dev/null ]; then
    echo >&2 "'waypipe' executable not found, install waypipe if you want to use remote debugging."
else
    rm -rf /tmp/squey-waypipe-socket-client && waypipe --socket /tmp/squey-waypipe-socket-client client &
fi

build_deps=1
while : ; do
    bst $TARGET_OPTION shell $MOUNT_OPTS --build squey.bst
    if [ $? -eq 255 ] && [ $build_deps -eq 1 ]; then
        DEPS=$(bst $TARGET_OPTION show --format=%{name} squey.bst)
        DEPS=$(echo "$DEPS" | tr '\n' ' ' | sed 's/ squey.bst//')
        bst $TARGET_OPTION build $DEPS
        build_deps=0
    else
        break;
    fi
done
