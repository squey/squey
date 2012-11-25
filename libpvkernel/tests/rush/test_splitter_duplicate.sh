#!/bin/bash

# \file test_splitter_duplicate.sh
#
# Copyright (C) Picviz Labs 2012

#

DIR="test-files/splitters/duplicate"

for f in $DIR/*.txt; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_duplicate "$INPUT" 6000 || (echo "Failed" && exit 1)
done
