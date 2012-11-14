#!/bin/bash

# \file make_protect
#
# Copyright (C) Picviz Labs 2010-2012

if [ $# -eq 2 ]; then
	echo "Protectiong for machine UUID $2..."
else
	echo "WARNING: no machine UUID given. Using the current machine's one..."
fi
BINDIR=@CMAKE_CURRENT_BINARY_DIR@
SRCDIR=@CMAKE_CURRENT_SOURCE_DIR@
echo "" >"$BINDIR/gui-qt/src/gen_table.c"
#CC=icecc && CXX=icecc cmake -DPROTECT_PASS=1 . ||exit $?
#cmake -DPROTECT_PASS=1 . ||exit $?
make -j$1

export CCACHE_DISABLE=1
touch "$SRCDIR/gui-qt/src/main.cpp"
make -j$1
make -j$1

make ||exit $?

cd "$BINDIR/gui-qt/src"
$GALVEZ_ROOT/tools/gen_table picviz-inspector $2 >gen_table.c ||exit $?
make ||exit $?
sh $GALVEZ_ROOT/bin/gen_patcher.sh gen_table.c ||exit $?
./patcher ./picviz-inspector
strip ./picviz-inspector
