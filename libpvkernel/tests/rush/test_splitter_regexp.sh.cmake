#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

#

ROOTDIR="@TESTS_FILES_DIR@"
DIR="$ROOTDIR/pvkernel/rush/splitters/regexp"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.regexp; do
	INPUT=${f%.*}
	REF=$INPUT.out
	REGEXP=$(cat $f)
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_splitter_regexp" "$INPUT" 6000 "$REGEXP" || (echo "Failed" && exit 1)
done
