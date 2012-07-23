#!/bin/bash

# \file test_splitter_csv.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

DIR="test-files/splitters/csv"

for f in $DIR/*.csv; do
	INPUT=$f
	REF=$f.out
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_splitter_csv "$INPUT" 6000 || (echo "Failed" && exit 1)
done
