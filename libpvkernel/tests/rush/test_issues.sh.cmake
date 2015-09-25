#!/bin/bash

# \file test_issues.sh
#
# Copyright (C) Picviz Labs 2010-2012

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
