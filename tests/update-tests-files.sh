#!/bin/bash
#
# @file
#
# 
# @copyright (C) ESI Group INENDI 2015-2015

if [ $# -ne 2 ]; then
	echo "Usage: $0 test-files-root rev-file" 1>&2
	exit 1
fi

FILES_DIR="$1"
FILE_REV="$2"

if [ ! -f "$FILE_REV" ]; then
	echo "File $FILE_REV does not exist." 1>&2
	exit 2
fi

REV=$(cat "$FILE_REV")

svn co https://svn.srv.picviz/tests-files@$REV "$FILES_DIR"

exit 0
