#!/bin/bash

# \file test_splitter_url.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_CURRENT_SOURCE_DIR@"
DIR="$ROOTDIR/test-files/splitters/url"

for f in $DIR/*.url; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_splitter_url" "$INPUT" 6000 || (echo "Failed" && exit 1)
done
