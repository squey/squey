#!/bin/sh

# \file update-headers.sh
#
# Copyright (C) Picviz Labs 2010-2012

##############################################################################
# USAGE:
# used with a pipe (as stdin), it will read file names from it
# with no parameter, it will process any files in git's index
# used with a parameter, it is a file containing the files to process
#

##############################################################################
# ABOUT
#
# This shell script process every file to update their headers:
# - if defined, it make it well-formed
# - if not, it create one
#

##############################################################################
# has_copyright $FILE
#
has_copyright()
{
    grep -q -i copyright "$1"
}

##############################################################################
# has_picviz_copyright $FILE
#
has_picviz_copyright()
{
    egrep -q -i '(tricaud|philippe|picviz)' "$1"
}

##############################################################################
# generic_split_with_old_header $FILE $HEADER $CONTENT [$BLANK_REGEXP] [TEXT_REGEXP]
#
generic_split_with_old_header()
{
    test $# -lt 3 -o $# -gt 5 && echo "bad parameters for generic_split_with_old_header" && exit 1
    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_HEADER="$2"
    LOCAL_CONTENT="$3"
    if test -z "$4"
    then
	BLANK_RE='^[[:blank:]]*$'
    else
	BLANK_RE="$4"
    fi
    if test -z "$5"
    then
	TEXT_RE='^.+$'
    else
	TEXT_RE="$5"
    fi
    FIND_COMMENT=
    FIND_BLANK=
    LINE_NUM=-1
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	if test -z "$FIND_COMMENT"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	    FIND_COMMENT=1
	elif test -z "$FIND_BLANK"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q "$TEXT_RE") && continue
	    FIND_BLANK=1
	else
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	    break
	fi
    done < "$LOCAL_FILE"

    # save header
    head -n $LINE_NUM < "$LOCAL_FILE" > "$LOCAL_HEADER"

    # save content
    LINE_NUM=`expr $LINE_NUM + 1`
    tail -n +$LINE_NUM < "$LOCAL_FILE" > "$LOCAL_CONTENT"
}

##############################################################################
# generic_split_with_no_header $FILE $CONTENT
#
generic_split_with_no_header()
{
    test $# -lt 2 -o $# -gt 3 && echo "bad parameters for generic_split_with_no_header" && exit 1
    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_CONTENT="$2"
    FIND_CONTENT=
    if test -z "$3"
    then
	BLANK_RE='^[[:blank:]]*$'
    else
	BLANK_RE="$3"
    fi
    LINE_NUM=0
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	(echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	break
    done < "$1"

    # save content
    tail -n +$LINE_NUM < "$LOCAL_FILE" > "$LOCAL_CONTENT"
}

##############################################################################
# cmake_split_with_old_header $FILE $HEADER $CONTENT [$BLANK_REGEXP] [TEXT_REGEXP]
#
cmake_split_with_old_header()
{
    test $# -lt 3 -o $# -gt 5 && echo "bad parameters for cmake_split_with_old_header" && exit 1
    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_HEADER="$2"
    LOCAL_CONTENT="$3"

    FIND_COPYRIGHT=
    FIND_BLANK=
    LINE_NUM=-1
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	if test -z "$FIND_COPYRIGHT"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -qvi "copyright") && continue
	    FIND_COPYRIGHT=1
	elif test -z "$FIND_BLANK"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -qi "copyright") && continue
	    FIND_BLANK=1
	else
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q '^#[[:blank:]]*$') && continue
	    break
	fi
    done < "$LOCAL_FILE"

    # save header
    head -n $LINE_NUM < "$LOCAL_FILE" > "$LOCAL_HEADER"

    # save content
    LINE_NUM=`expr $LINE_NUM + 1`
    tail -n +$LINE_NUM < "$LOCAL_FILE" > "$LOCAL_CONTENT"

}


##############################################################################
# html_split_with_old_header $FILE $HEADER $CONTENT [$BLANK_REGEXP] [TEXT_REGEXP]
#
html_split_with_old_header()
{
    test $# -lt 3 -o $# -gt 5 && echo "bad parameters for html_split_with_old_header" && exit 1
    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_HEADER="$2"
    LOCAL_CONTENT="$3"
    LINE_NUM=-1
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	(echo "$LINE" | sed -e 's/\r//' | egrep -q '^<!--') && continue
	break
    done < "$LOCAL_FILE"

    # save header
    head -n $LINE_NUM < "$LOCAL_FILE" > "$LOCAL_HEADER"

    # save content
    LINE_NUM=`expr $LINE_NUM + 1`
    tail -n +$LINE_NUM < "$LOCAL_FILE" > "$LOCAL_CONTENT"

}

##############################################################################
# main
##############################################################################

INPUT_LIST=

YEAR_DEFAULT_BEGIN=2010
YEAR_CURRENT=`date +"%Y"`

CURRENT_BRANCH=`git branch | colrm | grep '*' | cut -d\  -f 2`

# '("|)' in the next regexps takes care when git prints filenames with
# unicode characters

if test ! -t 0
then
    # reading from a pipe, dump it
    while read LINE
    do
	echo $LINE
    done
elif test -f "$1"
then
    # $1 is a file, dump it
    cat "$1"
else
    # list all files from git
    git ls-files
fi |
# some directories: test-files/, third-party/
egrep -v '(test-files|third-party|tests/filter/test-agg)/' |
# images, fonts, archives, "office" documents, Qt Help
egrep -v '\.(png|jpg|svg|bmp|ico|ttf|gz|bz2|zip|odt|ods|pdf|doc|docx|xls|xlsx|qhcp)("|)$' |
# Microsoft Visual Studio's files
egrep -v '\.(rc|qrc)("|)$' |
# relative Inspector files: .pv, .pcap, .log
egrep -v '\.(pv|pcap|log)("|)$' |
# ignore itself
egrep -v 'update-copyright.sh' |
while read FILE
do
    if test ! -s "$FILE"
    then
	echo "# file $FILE"
	echo "  ignored (empty file in branch $CURRENT_BRANCH)"
	continue
    fi


    CONTENT="$FILE.content"
    HAS_HEADER=

    if has_copyright "$FILE"
    then
	if has_picviz_copyright "$FILE"
	then
	    # file owned by Picviz Labs
	    HAS_HEADER=1
	else
	    # file not owned by Picviz Labs
	    # nothing to do
	    continue
	fi
    else
	# file owned by Picviz Labs but without any copyright
	HAS_HEADER=0
    fi

    FILENAME=`basename "$FILE"`

    case "$FILE" in
	cmake/FindAPR.cmake | cmake/FindBoost.cmake | cmake/FindDUMBNET.cmake | cmake/FindHDFS.cmake | cmake/FindHWLoc.cmake | cmake/FindICU.cmake | cmake/FindPCAP.cmake | cmake/FindPCRE.cmake | cmake/FindPkgMacros.cmake | cmake/FindTBB.cmake | cmake/FindTULIP3.cmake | cmake/UseJavaClassFilelist.cmake | cmake/UseJava.cmake | cmake/UseJavaSymlinks.cmake | libpicviz/src/include/picviz/PVBitset_gnu.h )
	    # files not owned by Picviz Labs
	    ;;

	*CHANGELOG | scripts/versions/versions | libpicviz/doc/LICENSES.txt | COPYING.txt | doc/license-demo.txt | pvconfig.ini | *syslog | VERSION.txt | VERSION-NAME.txt | CMakeCustomerSpecifics.txt | CMakePicvizDeveloperSpecifics.txt )
	    # files which do not need any header
	    ;;

	*.csv )
	    # files whose format do not define any comment syntax
	    ;;

	libpicviz/src/include/picviz/PVBitset_gnu.h | libpvkernel/plugins/input_types/database/mysql_types.h | */uchardetect/* )
	    # files to ignore
	    ;;

	libpvkernel/tests/rush/test-logs/csv/tiny.csv.format )
	    echo "# file $FILE"
	    echo "  strange format"
	    ;;

	*.c | *.h | *.cpp | *.hpp | *.cxx | *.hxx | *.java | *.geom | *.vert | *.frag | *.cu | include/pvbase/version.h.cmake | *.css )
	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	generic_split_with_old_header "$FILE" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
/**
 * \file $FILENAME
 *
 * Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT
 */

EOF
	    cat "$CONTENT" >> "$FILE"
	    ;;

	*.txt | *Doxyfile* | *README | libpicviz/CTestConfig.cmake | doc/FindPICVIZ.cmake | picviz-tools/textdig/cmake/FindPICVIZ.cmake )
	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	generic_split_with_old_header "$FILE" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
#
# \file $FILENAME
#
# Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT

EOF
	    cat "$CONTENT" >> "$FILE"
	    ;;

	*.sh | make_protect | parallel_make | doc/gen_doc | *.py | *.pl )
	    SHEBANG=`head -n 1 "$FILE"`
	    FILE_TMP="$FILE.tmp"
	    tail -n +2 < "$FILE" > "$FILE_TMP"

	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	generic_split_with_old_header "$FILE_TMP" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE_TMP" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
$SHEBANG

# \file $FILENAME
#
# Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT

EOF
	    cat "$CONTENT" >> "$FILE"
	    rm -f "$FILE_TMP"
	    ;;

	*.bat )
	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	generic_split_with_old_header "$FILE" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
REM
REM \file $FILENAME
REM
REM Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT

EOF
	    cat "$CONTENT" >> "$FILE"
	    ;;

	cmake/* )
	    INFO=`head -n 1 "$FILE"`
	    FILE_TMP="$FILE.tmp"
	    tail -n +2 < "$FILE" > "$FILE_TMP"

	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	cmake_split_with_old_header "$FILE_TMP" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE_TMP" "$CONTENT" '^#?[[:blank:]]*$'
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
$INFO
#
# \file $FILENAME
#
# Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT
#
EOF
	    cat "$CONTENT" >> "$FILE"
	    rm -f "$FILE_TMP"
	    ;;

	*.html )
	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	html_split_with_old_header "$FILE" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
<!-- \file $FILENAME -->
<!-- Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT -->
EOF
	    cat "$CONTENT" >> "$FILE"
	    ;;

	*.format | *.ui )
	    INFO=`head -n 1 "$FILE"`
	    FILE_TMP="$FILE.tmp"
	    tail -n +2 < "$FILE" > "$FILE_TMP"

	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	html_split_with_old_header "$FILE_TMP" "$HEADER" "$CONTENT"
		# extract the earlier copyright year in old header
		YEAR_BEGIN=`cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1`
		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE_TMP" "$CONTENT"
		YEAR_BEGIN=$YEAR_DEFAULT_BEGIN
	    fi
	    cat > "$FILE" <<EOF
$INFO
<!-- \file $FILENAME -->
<!-- Copyright (C) Picviz Labs ${YEAR_BEGIN}-$YEAR_CURRENT -->
EOF
	    cat "$CONTENT" >> "$FILE"
   	    rm -f "$FILE_TMP"
	    ;;

	* )
	    echo "# file $FILE"
	    echo "  which action?"
	    ;;
    esac

    rm -f "$CONTENT"
done
