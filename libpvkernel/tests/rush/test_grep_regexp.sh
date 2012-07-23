#!/bin/bash

# \file test_grep_regexp.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

DIR="test-files/grep/regexp"

for f in $DIR/*.regexp; do
	INPUT=${f%.*}
	REF=$INPUT.out
	REGEXP=$(cat $f)
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_grep_regexp "$INPUT" 6000 "$REGEXP" || (echo "Failed" && exit 1)
done
