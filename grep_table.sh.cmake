#!/bin/sh

BINDIR=@CMAKE_CURRENT_BINARY_DIR@
SRCDIR=@CMAKE_CURRENT_SOURCE_DIR@

STR=

while test -n "$1"
do
    STR="${STR}$1"
    shift
done

grep "$STR" "$BINDIR/gen_table.c"

