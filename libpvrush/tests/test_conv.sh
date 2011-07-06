#!/bin/bash
# Test UTF16 conversion
# Usage: ./test_conv.sh charset
# The file libpvrush/tests/charset/$charset must exists. Diff output will
# go to libpvrush/tests/charset/$charset.diff

if [ $# -ne 1 ]; then
	echo "Usage: $0 charset" 1>&2
	exit 1
fi

INPUT="./test-files/charset/$1"
DIFF="$INPUT.diff"
OUT="$INPUT.out"

./diff_stdout.py "$OUT" "$DIFF" ./Trush_conv_utf16 "$INPUT" 20000
exit $?
