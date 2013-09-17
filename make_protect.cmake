#!/bin/bash

# \file make_protect
#
# Copyright (C) Picviz Labs 2010-2012

BINDIR=`pwd`

if [ $# -eq 2 ]; then
	echo "Protectiong for machine UUID $2..."
else
	echo "WARNING: no machine UUID given. Using the current machine's one..."
fi
BINDIR=@CMAKE_CURRENT_BINARY_DIR@
SRCDIR=@CMAKE_CURRENT_SOURCE_DIR@

# first we need an "empty" gen_table.c
#
cat << EOF >"$BINDIR/gui-qt/src/gen_table.c"
#include <stdint.h>
#include <stdlib.h>
#include <fwd_table.h>

f_entry_t __func_table[] = {};

size_t __func_table_size = 0;
EOF

# compilation with mao (and make sure there is no error)
#
#CC=icecc && CXX=icecc cmake -DPROTECT_PASS=1 . ||exit $?
#cmake -DPROTECT_PASS=1 . ||exit $?
make -j$1

cp picviz-inspector picviz-inspector.with-empty-table

export CCACHE_DISABLE=1
touch "$SRCDIR/gui-qt/src/main.cpp"
make -j$1
make -j$1

make ||exit $?

# finalizing protection
cd "$BINDIR/gui-qt/src"
TFILE="./dump"
objdump -d picviz-inspector | grep '>:$' > "$TFILE"
$GALVEZ_ROOT/tools/gen_table picviz-inspector $2 > gen_table.c ||exit $?

cp gen_table.c gen_table.old.c
while read LINE
do
	if expr match "$LINE" "^{{.*" 2>&1 > /dev/null
	then
		# reading address field
	        ADDR=`echo "$LINE" | sed -e 's+{{[^}]*}, [^0]*0x\([^,]*\).*+\1+'`
		# a symbol has to be search for only if its address is not NULL
		if test $ADDR -ne 0
		then
			# searching for the symbol corresponding to this address
		        FUNC=`grep "$ADDR" "$TFILE" | sed -e 's+[^<]*<\([^>]*\)>:$+\1+'`
			# removing the '*/' from the line
		        LINE=`echo "$LINE" | sed -e 's+^\(.*\)\*/$+\1+'`
			# adding the symbol name to the comment (and closing it:)
			LINE="$LINE FUNC=$FUNC */"
		fi
	fi
	echo "$LINE"
done < gen_table.old.c > gen_table.new.c

cp gen_table.new.c gen_table.c

cp gen_table.new.c "$BINDIR/gen_table.c"

# final compilation
#
make ||exit $?

cp picviz-inspector picviz-inspector.unpatched-with-table

sh $GALVEZ_ROOT/bin/gen_patcher.sh gen_table.c ||exit $?
./patcher ./picviz-inspector

cp ./picviz-inspector picviz-inspector.with-debug

# getting debug info
objcopy --only-keep-debug ./picviz-inspector "$BINDIR/picviz-inspector.symbols"
cp ./picviz-inspector "$BINDIR/picviz-inspector-with-debug"

cp ./picviz-inspector picviz-inspector.after-objcopy

# saving a stripped version of PV-I
strip --strip-all "$BINDIR/gui-qt/src/picviz-inspector" -o "$BINDIR/gui-qt/src/picviz-inspector.stripped"

