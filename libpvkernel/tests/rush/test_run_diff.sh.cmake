#!/bin/bash

# \file test_run_diff.sh.cmake
#
# Copyright (C) Picviz Labs 2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpvkernel/tests/rush"

test $# -eq 0 && echo "usage: `basename $0` [parameters for diff_stdout.py]" && exit 1

PARAMS=""

while test ! -z "$1"
do
    case "$1" in
	test-files/*)
	    PARAMS="$PARAMS \"$ROOTDIR\"/$1"
	    ;;
	*)
	    PARAMS="$PARAMS $1"
	    ;;
    esac
    shift
done

"$ROOTDIR"/diff_stdout.py $PARAMS
