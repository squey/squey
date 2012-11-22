#!/bin/bash

# \file squish_run.sh.cmake
#
# Copyright (C) Picviz Labs 2012

[[ -n "$1" ]] || { echo "Usage: `basename $0` <test_suite> "; exit 0 ; }

export INSPECTOR_SOURCE_DIR=@CMAKE_SOURCE_DIR@
export INSPECTOR_BUILD_DIR=@CMAKE_BINARY_DIR@
export INSPECTOR_TESTS_DIR="$INSPECTOR_SOURCE_DIR/tests"
export SQUISH_DIR="$INSPECTOR_TESTS_DIR/squish"
export SQUISH_SCRIPT_DIR="$SQUISH_DIR/scripts"

CMD_ARGS=("$@")

cd "$SQUISH_DIR/test_suites"

squishserver --daemon
squishserver --config addAUT inspector.sh $INSPECTOR_SOURCE_DIR 
squishrunner --testsuite ${CMD_ARGS[@]}