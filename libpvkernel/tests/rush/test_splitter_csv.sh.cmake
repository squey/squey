#!/bin/bash

# \file test_splitter_csv.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_CURRENT_SOURCE_DIR@"
DIR="$ROOTDIR/test-files/splitters/csv"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.csv; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_splitter_csv" "$INPUT" 6000 || (echo "Failed" && exit 1)
done
