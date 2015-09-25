#!/bin/bash

# \file test_splitter_pcap.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@TESTS_FILES_DIR@"
DIR="$ROOTDIR/pvkernel/rush/splitters/pcap"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.pcap; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"@TEST_DIFF_STDOUT@" "$REF" "$INPUT.diff" "@CMAKE_CURRENT_BINARY_DIR@/Trush_splitter_pcap" "$INPUT" 6000 || (echo "Failed" && exit 1)
done
