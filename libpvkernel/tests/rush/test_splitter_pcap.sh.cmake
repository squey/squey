#!/bin/bash

# \file test_splitter_pcap.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

ROOTDIR="@CMAKE_SOURCE_DIR@/libpvkernel/tests/rush"
DIR="$ROOTDIR/test-files/splitters/pcap"

test ! -d "$DIR" && echo "'$DIR' is not a directory" && exit 1

for f in $DIR/*.pcap; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	"$ROOTDIR"/diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_pcap "$INPUT" 6000 || (echo "Failed" && exit 1)
done
