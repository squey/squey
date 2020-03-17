#!/bin/bash

set -e

source .common.sh
scripts/cuda-devices &> /dev/null || true

open_workspace "workspace_dev"

bst shell $MOUNT_OPTS --build inendi-inspector.bst
