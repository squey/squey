#!/bin/sh

##############################################################################
# returns the relative path of $1 against $2
#
# example: the call to:
# $ get_relative_path /tests/sources/test-petit.log /tests/outputs/0
# will print:
# ../../sources/test-petit.log
#
get_relative_path()
{
    test ! -f "$1" && echo "get_relative_path: '$1' is not reachable" 1>&2 && exit 1
    test ! -d "$2" && echo "get_relative_path: '$2' is not reachable" 1>&2 && exit 1

    FILENAME=`realpath "$1"`
    COMMON_PATH=`dirname $FILENAME`
    REFDIR=`realpath "$2"`

    # the goal is to have the common path of $1 and $2 in COMMON_PATH
    while true
    do
	expr "$REFDIR" : "$COMMON_PATH" 2>&1 > /dev/null
	if test $? -ne 0
	then
	    COMMON_PATH=`dirname "$COMMON_PATH"`
	else
	    break
	fi
    done

    # P1 is the ascendant part in the path from $2 to $1
    P1=`echo "$REFDIR" | sed -e "s|$COMMON_PATH/||"  -e 's|[^/]*|\.\.|g'`

    # P2 is the descendant part in the path from $2 to $1
    P2=`echo "$FILENAME" | sed -e "s|$COMMON_PATH/||"`
    FN="$P1/$P2"

    # check that the file is accessible from $2
    D=`pwd`
    cd "$2"
    test ! -r "$FN" && echo "get_relative_path: bad result!" 1>&2 && echo && exit 2
    cd "$D"

    echo $FN
}

##############################################################################
# env
#
PNAME=`basename $0`

PI_FILES_DIR=`realpath $0 | xargs dirname | xargs dirname`
PI_SOURCE_DIR=`dirname "$PI_FILES_DIR"`

PI_FILES_DIR="$PI_FILES_DIR/files"
PI_OUTPUTS_DIR="$PI_FILES_DIR/outputs"

PI_MAPPING_DUMPER="$PI_SOURCE_DIR/libpicviz/tests/bin/Tpicviz_process_file_mapping"
test ! -x "$PI_MAPPING_DUMPER" && echo "'$PI_MAPPING_DUMPER' not reachable" 1>&2 && exit 2

PI_PLOTTING_DUMPER="$PI_SOURCE_DIR/libpicviz/tests/bin/Tpicviz_process_file_plotting"
test ! -x "$PI_PLOTTING_DUMPER" && echo "'$PI_PLOTTING_DUMPER' not reachable" 1>&2 && exit 2

PI_CSV_DUMPER="$PI_SOURCE_DIR/libpvkernel/tests/rush/bin/Trush_process_file"
test ! -x "$PI_CSV_DUMPER" && echo "'$PI_CSV_DUMPER' not reachable" 1>&2 && exit 2

PI_ZT_DUMPER="$PI_SOURCE_DIR/libpvparallelview/tests/bin/Tpview_dump_zone_tree"
test ! -x "$PI_ZT_DUMPER" && echo "'$PI_ZT_DUMPER' not reachable" 1>&2 && exit 2

PI_ZZT_DUMPER="$PI_SOURCE_DIR/libpvparallelview/tests/bin/Tpview_dump_zoomed_zone_tree"
test ! -x "$PI_ZZT_DUMPER" && echo "'$PI_ZZT_DUMPER' not reachable" 1>&2 && exit 2

##############################################################################
# main
# $1 = input-file
# $2 = input-format
#
test $# -ne 2 && echo "usage: `basename $0` input-file input-format" && exit 1

test ! -r "$1" && echo "'$1' is not a file or is not readable" && exit 1
test ! -r "$2" && echo "'$2' is not a file or is not readable" && exit 1

REAL_INPUT_FILE=`realpath "$1"`
REAL_INPUT_FILE_NAME=`basename "$REAL_INPUT_FILE"`
REAL_INPUT_FORMAT=`realpath "$2"`

# Check if a test-case already use these 2 files. It's done by finding
# test-cases which use the same input file and to tests their format file.
#
FOUND_FILE="/tmp/$PNAME.found.$$"

# make sure it does not exist
rm -f "$FOUND_FILE"

find "$PI_FILES_DIR/outputs" -name "input-file" | while read IFILE
do
    IDIR=`dirname "$IFILE"`
    REALNAME=`cd "$IDIR"; readlink -n input-file | xargs realpath`
    test "$REALNAME" = "$REAL_INPUT_FILE" && dirname "$IFILE"
done | while read IDIR
do
    # testing all test-case to see if they use the same input format
    REALNAME=`cd "$IDIR" ; readlink -n input-format | xargs realpath`
    if test "$REALNAME" = "$REAL_INPUT_FORMAT"
    then
	echo "$IDIR" > "$FOUND_FILE"
    fi
done

if test -s "$FOUND_FILE"
then
    echo "this test-case already exists in '`cat "$FOUND_FILE"`'"
    rm -f "$FOUND_FILE"
    exit 0
fi

# time to add a new test-case \o/
#

# finding the first available number
#
TMPFILE="/tmp/$PNAME.next_entries.$$"

(
    NEXT=`ls -1 "$PI_OUTPUTS_DIR" | tail -1`
    if test -n "$NEXT"
    then
	NEXT=`echo "$NEXT + 1" | bc`
    else
	NEXT=0
    fi
    ls -1 "$PI_OUTPUTS_DIR"
    seq 0 $NEXT
) | sort -n | uniq -u > "$TMPFILE"

NEXT=`tail -1 "$TMPFILE"`
rm -f "$TMPFILE"

NEXTDIR="$PI_OUTPUTS_DIR/$NEXT"
mkdir "$NEXTDIR"

INPUT_FILE=`get_relative_path "$1" "$NEXTDIR"`
INPUT_FORMAT=`get_relative_path "$2" "$NEXTDIR"`

test -z "$INPUT_FILE" && exit 1
test -z "$INPUT_FORMAT" && exit 1

cd "$NEXTDIR"

ln -s "$INPUT_FILE" input-file
ln -s "$INPUT_FORMAT" input-format

echo "generating mapped file"
$PI_MAPPING_DUMPER input-file input-format > mapped 2> /dev/null
if test $? -ne 0
then
    echo "error generating mapped file; check, remove, and rerun" 1>&2
    echo "commands:"
    echo "cd \"$NEXTDIR\""
    echo "$PI_MAPPING_DUMPER input-file input-format > mapped"
    exit 1
fi

echo "generating plotted file"
$PI_PLOTTING_DUMPER input-file input-format 1 1 plotted 2> /dev/null
if test $? -ne 0
then
    echo "error generating plotted file; check, remove, and rerun" 1>&2
    echo "commands:"
    echo "cd \"$NEXTDIR\""
    echo "$PI_PLOTTING_DUMPER input-file input-format 1 1 plotted"
    exit 1
fi

echo "generating CSV file"
$PI_CSV_DUMPER input-file input-format > csv 2> /dev/null
if test $? -ne 0
then
    echo "error generating CSV file; check, remove, and rerun" 1>&2
    echo "commands:"
    echo "cd \"$NEXTDIR\""
    echo "$PI_CSV_DUMPER input-file input-format > csv"
    exit 1
fi

echo "generating zone trees file"
$PI_ZT_DUMPER plotted 1 "zt-%d" all > /dev/null 2>&1
if test $? -ne 0
then
    echo "error generating zone trees files; check, remove, and rerun" 1>&2
    echo "commands:"
    echo "cd \"$NEXTDIR\""
    echo "$PI_ZT_DUMPER plotted 1 \"zzt-%d\" all"
    exit 1
fi

echo "generating zoomed zone trees files"
$PI_ZZT_DUMPER plotted 1 "zzt-%d" all > /dev/null 2>&1
if test $? -ne 0
then
    echo "error generating zoomed zone trees files; check, remove, and rerun" 1>&2
    echo "commands:"
    echo "cd \"$NEXTDIR\""
    echo "$PI_ZZT_DUMPER plotted 1 \"zzt-%d\" all"
    exit 1
fi
