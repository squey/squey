#!/bin/bash

# \file test_issues.sh
#
# Copyright (C) Picviz Labs 2010-2012

#

DIR="test-files/had_issues/$1"

for f in $DIR/*.format; do
	INPUT=${f%.*}
	REF=$INPUT.out
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Tpicviz_process_file_$1 "$INPUT" "$f" || (echo "Failed" && exit 1)
done
