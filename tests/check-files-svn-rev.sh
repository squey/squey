#!/bin/bash
#

if [ $# -ne 2 ]; then
	echo "Usage: $0 test-files-root rev-file" 1>&2
	exit 1
fi

SCRIPT_DIR=$(/usr/bin/realpath $(dirname $0))
FILES_DIR=$1
FILE_REV=$2

if [ ! -f "$FILE_REV" ]; then
	echo "File $FILE_REV does not exist." 1>&2
	exit 1
fi

if [ ! -d "$FILES_DIR" ]; then
	echo "Directory $FILES_DIR does not exist." 1>&2
	exit 1
fi

REV=$(cd "$FILES_DIR"; /usr/bin/svnversion)

if echo $REV | egrep -q -v '^[0-9]*:[0-9]*$'; then
	echo "Unable to get SVN revision from $FILES_DIR" 1>&2
	exit 1
fi
CMP_REV=$(cat "$FILE_REV")

if [ "$REV" -ne "$CMP_REV" ]; then
	echo "Test files are at the revision $REV, and $CMP_REV is needed (as specified by $FILE_REV). Please update your SVN repository according to this revision." 1>&2
	exit 1
fi


exit 0
