#!/bin/bash

# \file test_splitter_regexp.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

DIR="test-files/splitters/regexp"

for f in $DIR/*.regexp; do
	INPUT=${f%.*}
	REF=$INPUT.out
	REGEXP=$(cat $f)
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_regexp "$INPUT" 6000 "$REGEXP" || (echo "Failed" && exit 1)
done
