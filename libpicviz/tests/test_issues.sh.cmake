#!/bin/bash

# \file test_issues.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpicviz/tests"
DIR="$ROOTDIR/test-files/had_issues/$1"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.format; do
	INPUT=${f%.*}
	REF=$INPUT.out
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Tpicviz_process_file_$1 "$INPUT" "$f" || (echo "Failed" && exit 1)
done
