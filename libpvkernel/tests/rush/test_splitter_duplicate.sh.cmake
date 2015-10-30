#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2012-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

#

ROOTDIR="@TESTS_FILES_DIR@"
DIR="$ROOTDIR/pvkernel/rush/splitters/duplicate"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.txt; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_splitter_duplicate" "$INPUT" 6000 || (echo "Failed" && exit 1)
done
