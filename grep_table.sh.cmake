#!/bin/sh
#
# @file
#
# @copyright (C) Picviz Labs 2013-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

BINDIR=@CMAKE_CURRENT_BINARY_DIR@
SRCDIR=@CMAKE_CURRENT_SOURCE_DIR@

STR=

while test -n "$1"
do
    STR="${STR}$1"
    shift
done

grep "$STR" "$BINDIR/gen_table.c"

