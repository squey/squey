#!/bin/bash

set -e

xhost +

source .common.sh
scripts/cuda-devices &> /dev/null || true

bst build base.bst
bst shell $MOUNT_OPTS --build squey.bst
