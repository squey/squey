#!/bin/sh
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

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
# has_our_copyright $FILE
#
has_our_copyright()
{
    egrep -q -i '(tricaud|philippe|picviz|inendi)' "$1"
}

##############################################################################
# generic_split_with_old_header $FILE $HEADER $CONTENT $INFO $INFO_LINE [TEXT_RE]
#
# Read file and print header part in $HEADER file and content part in $CONTENT file
# $INFO_LINE first lines are skipped and saved in $INFO file
generic_split_with_old_header()
{
    test $# -lt 5 && echo "bad parameters for generic_split_with_old_header" && exit 1

    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_HEADER="$2"
    LOCAL_CONTENT="$3"
    LOCAL_INFO="$4"
    INFO_LINE="$5"

    # Define separator pre and post copyright
    BLANK_RE='^[[:blank:]]*$'

    # Define copyright part definition
    if $(echo "$FILE" | egrep -q "(.format|.ui|.html|.qrc)$")
    then
	TEXT_RE='^<!--'
    else
	TEXT_RE='^.+$'
    fi

    FIND_COMMENT=
    FIND_BLANK=
    LINE_NUM=-1
    # Save, first lines. Read all empty lines, then all text lines and finally
    # all empty lines. All of these lines are header part.
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	if test "$LINE_NUM" -lt "$INFO_LINE"
	then
	    continue
	fi
	if test -z "$FIND_COMMENT"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	    FIND_COMMENT=1
	fi
	if test -z "$FIND_BLANK"
	then
	    (echo "$LINE" | sed -e 's/\r//' | egrep -q "$TEXT_RE") && continue
	    FIND_BLANK=1
	fi
	(echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	break
    done < "$LOCAL_FILE"

    # save info
    head -n $INFO_LINE < "$LOCAL_FILE" > "$LOCAL_INFO"

    # save header
    head -n $LINE_NUM < "$LOCAL_FILE" | tail -n +$INFO_LINE > "$LOCAL_HEADER"

    # save content
    LINE_NUM=`expr $LINE_NUM + 1`
    tail -n +$LINE_NUM < "$LOCAL_FILE" > "$LOCAL_CONTENT"
}

##############################################################################
# generic_split_with_no_header $FILE $CONTENT $INFO $INFO_LINE
#
# Read file and print content part in $CONTENT file
# Remove first empty lines
generic_split_with_no_header()
{
    test $# -lt 4 && echo "bad parameters for generic_split_with_no_header" && exit 1
    # search for the first line of relevant code
    LOCAL_FILE="$1"
    LOCAL_CONTENT="$2"
    LOCAL_INFO="$3"
    INFO_LINE=$4
    FIND_CONTENT=
    BLANK_RE='^[[:blank:]]*$'
    LINE_NUM=-1
    while read LINE
    do
	LINE_NUM=`expr $LINE_NUM + 1`
	if test "$LINE_NUM" -lt "$INFO_LINE"
	then
	    continue;
	fi
	(echo "$LINE" | sed -e 's/\r//' | egrep -q "$BLANK_RE") && continue
	break
    done < "$LOCAL_FILE"

    # save info
    head -n $LINE_NUM < "$LOCAL_FILE" > "$LOCAL_INFO"

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

###########################################################
# Inputs are filenames.
#
# Filenames may be from stdin or listed in a file
# Without arguments, all git tracked files will be handled
###########################################################
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
# images, fonts, archives, "office" documents, Qt Help, conf
egrep -v '\.(png|jpg|svg|bmp|ico|ttf|gz|bz2|zip|odt|ods|pdf|doc|docx|xls|xlsx|qhcp|glif|fea|glyph|gif|plist|xcf|conf|mm|woff)("|)$' |
# relative Inspector files: .pv, .pcap, .log
egrep -v '\.(pv|pcap|log)("|)$' |

###################################
# Process each files
###################################
while read FILE
do
    if test ! -s "$FILE"
    then
	echo "# file $FILE"
	echo "  ignored (empty file in branch $CURRENT_BRANCH)"
	continue
    fi


    CONTENT="$FILE.content"
    INFO="$FILE.info"
    HAS_HEADER=

    if has_copyright "$FILE"
    then
	if has_our_copyright "$FILE"
	then
	    # file owned by Us
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

    case "$FILE" in
	cmake/FindAPR.cmake | cmake/FindBoost.cmake | cmake/FindDUMBNET.cmake | cmake/FindHWLoc.cmake | cmake/FindICU.cmake | cmake/FindPCRE.cmake | cmake/FindPkgMacros.cmake | cmake/FindTBB.cmake | *.js)
	    # files not owned by us
	    ;;

	*asciidoc | *CHANGELOG | scripts/versions/versions | libinendi/doc/LICENSES.txt | COPYING.txt | doc/license-demo.txt | pvconfig.ini | *syslog | VERSION.txt | VERSION-NAME.txt | CMakeCustomerSpecifics.txt | CMakePicvizDeveloperSpecifics.txt | .gitignore | *.graphml | *files-svn-rev)
	    # files which do not need any header
	    ;;

	*.csv )
	    # files whose format do not define any comment syntax
	    ;;

	libpvkernel/plugins/input_types/database/mysql_types.h | */uchardetect/* )
	    # files to ignore
	    ;;

	* )
	    case "$FILE" in
		ln_build | *.sh | *.sh.cmake | parallel_make | doc/gen_doc | *.py | *.pl | *.format | *.ui )
		    INFO_LINE=1
		    ;;
		*.c | *.h | *.h.cmake | *.cpp | *.hpp | *.cxx | *.hxx | *.java | *.geom | *.vert | *.frag | *.cu | *.css | *.txt | *Doxyfile* | *README | libinendi/CTestConfig.cmake | cmake/* | *.html | *.qrc | *.obj)
		    INFO_LINE=0
		    ;;
		* )
		    echo "UNKNOWN FILE $FILE";
	    esac

	    if test $HAS_HEADER -ne 0
	    then
		HEADER="$FILE.header"
	    	generic_split_with_old_header "$FILE" "$HEADER" "$CONTENT" "$INFO" "$INFO_LINE"
		# extract the earlier copyright year in old header
		# As it is looking for the first copyright, it is picviz if it was already present or INENDI
		YEAR_BEGIN=$(cat $HEADER | grep -i copyright | sed -e 's/-/ /g' | grep -o -E '\w+' | egrep '[0-9]{4}' | sort | head -n 1)
		GIT_YEAR_BEGIN=$(git log --format=%ad --date=short --follow $FILE | tail -1 | cut -c1-4)

		# Get the min year between git information and previous copyright information
		YEAR_BEGIN=$( [ $YEAR_BEGIN -le $GIT_YEAR_BEGIN ] && echo "$YEAR_BEGIN" || echo "$GIT_YEAR_BEGIN" )

		rm -f "$HEADER"
	    else
		generic_split_with_no_header "$FILE" "$CONTENT" "$INFO" "$INFO_LINE"
		YEAR_BEGIN=$(git log --format=%ad --date=short --follow $FILE | tail -1 | cut -c1-4)
	    fi
	    # Picviz copyright stop in 2014
	    if [ $YEAR_BEGIN -le 2014 ]
	    then
		PICVIZ_COPYRIGHT="@copyright (C) Picviz Labs ${YEAR_BEGIN}-March 2015"
	    else
		PICVIZ_COPYRIGHT=""
	    fi
	    # INENDI Copyright start in 2015
	    if [ $YEAR_BEGIN -gt 2014 ]
	    then
		YEAR_BEGIN=${YEAR_BEGIN}
	    else
		YEAR_BEGIN="April 2015"
	    fi
	    cat "$INFO" > "$FILE"

	    case "$FILE" in
		*.c | *.h | *.cpp | *.hpp | *.cxx | *.hxx | *.java | *.geom | *.vert | *.frag | *.cu | *.h.cmake | *.css )
		    cat >> "$FILE" <<EOF
/**
 * @file
 *
 * ${PICVIZ_COPYRIGHT}
 * @copyright (C) ESI Group INENDI ${YEAR_BEGIN}-$YEAR_CURRENT
 */

EOF
		    cat "$CONTENT" >> "$FILE"
		    ;;

		*.txt | *Doxyfile* | *README | libinendi/CTestConfig.cmake | cmake/* | *.sh | ln_build | *.sh.cmake | parallel_make | doc/gen_doc | *.py | *.pl | *.obj)
		    cat >> "$FILE" <<EOF
#
# @file
#
# ${PICVIZ_COPYRIGHT}
# @copyright (C) ESI Group INENDI ${YEAR_BEGIN}-$YEAR_CURRENT

EOF
		    cat "$CONTENT" >> "$FILE"
		    ;;

		*.html | *.format | *.ui | *.qrc )
		    cat >> "$FILE" <<EOF
<!-- @file -->
<!-- ${PICVIZ_COPYRIGHT} -->
<!-- @copyright (C) ESI Group INENDI ${YEAR_BEGIN}-$YEAR_CURRENT -->
EOF
		    cat "$CONTENT" >> "$FILE"
		    ;;

		* )
		    echo "# file $FILE"
		    echo "  which action?"
		    ;;
	    esac
	;;
    esac

    rm -f "$CONTENT"
    rm -f "$INFO"
done
