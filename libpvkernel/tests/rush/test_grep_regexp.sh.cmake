#!/bin/bash

# \file test_grep_regexp.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_CURRENT_SOURCE_DIR@"
DIR="$ROOTDIR/test-files/grep/regexp"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.regexp; do
	INPUT=${f%.*}
	REF=$INPUT.out
	REGEXP=$(cat $f)
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_grep_regexp" "$INPUT" 6000 "$REGEXP" || (echo "Failed" && exit 1)
done
