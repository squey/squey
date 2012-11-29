#!/bin/bash

# \file test_conv.sh
#
# Copyright (C) Picviz Labs 2010-2012

# Test UTF16 conversion
# Usage: ./test_conv.sh charset
# The file libpvrush/tests/charset/$charset must exists. Diff output will
# go to libpvrush/tests/charset/$charset.diff

if [ $# -ne 1 ]; then
	echo "Usage: $0 charset" 1>&2
	exit 1
fi

INPUT=$1
DIFF="$INPUT.diff"
OUT="$INPUT.out"
SCRIPT_PATH=$(/usr/bin/realpath $(dirname $0))

"$SCRIPT_PATH/diff_stdout.py" "$OUT" "$DIFF" ./Trush_conv_utf16 "$INPUT" 20000
exit $?
