#!/bin/bash

set -e

xhost +

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

echo "Activating python virtual environment"
source .venv/bin/activate
pip install --upgrade pip

source .common.sh
scripts/cuda-devices &> /dev/null || true

bst build base.bst
bst shell $MOUNT_OPTS --build squey.bst
