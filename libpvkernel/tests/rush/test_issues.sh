#!/bin/bash
#

DIR="test-files/had_issues"

for f in $DIR/*.format; do
	INPUT=${f%.*}
	REF=$INPUT.out
	echo "Testing $INPUT..."
	./diff_stdout.py "$REF" "$INPUT.diff" ./Trush_process_file "$INPUT" "$f" || (echo "Failed" && exit 1)
done
