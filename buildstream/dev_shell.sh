#!/bin/bash

set -e

xhost +

source .common.sh
scripts/cuda-devices &> /dev/null || true

bst build base.bst
if [ command -v "waypipe" &> /dev/null ]; then
    echo >&2 "'waypipe' executable not found, install waypipe if you want to use remote debugging."
else
    rm -rf /tmp/squey-waypipe-socket-client && waypipe --socket /tmp/squey-waypipe-socket-client client &
fi

bst shell $MOUNT_OPTS --build squey.bst
