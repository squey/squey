#!/bin/bash

# \file test_splitter_url.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpvkernel/tests/rush"
DIR="$ROOTDIR/test-files/splitters/url"

for f in $DIR/*.url; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_url "$INPUT" 6000 || (echo "Failed" && exit 1)
done
