#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

#

ROOTDIR="@TESTS_FILES_DIR@"
DIR="$ROOTDIR/pvkernel/rush/had_issues"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.format; do
	INPUT=${f%.*}
	REF=$INPUT.out
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Trush_process_file "$INPUT" "$f" || (echo "Failed" && exit 1)
done
