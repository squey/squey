#!/bin/bash

# \file test_splitter_regexp.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpvkernel/tests/rush"
DIR="$ROOTDIR/test-files/splitters/regexp"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.regexp; do
	INPUT=${f%.*}
	REF=$INPUT.out
	REGEXP=$(cat $f)
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_regexp "$INPUT" 6000 "$REGEXP" || (echo "Failed" && exit 1)
done
