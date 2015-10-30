#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2012-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

#

ROOTDIR="@TESTS_FILES_DIR@/pvkernel/rush"

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
