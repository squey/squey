#!/bin/bash

set -e

xhost +

source .common.sh
scripts/cuda-devices &> /dev/null || true

if [ -d ".venv" ]; then
    echo "Activating python virtual environment"
    source .venv/bin/activate
fi

bst shell $MOUNT_OPTS --build squey.bst
