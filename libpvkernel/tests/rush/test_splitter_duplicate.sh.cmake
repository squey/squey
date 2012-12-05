#!/bin/bash

# \file test_splitter_duplicate.sh
#
# Copyright (C) Picviz Labs 2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpvkernel/tests/rush"
DIR="$ROOTDIR/test-files/splitters/duplicate"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.txt; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_duplicate "$INPUT" 6000 || (echo "Failed" && exit 1)
done
