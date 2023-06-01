#!/bin/bash
#
# @file
#

ISRD=@CMAKE_SOURCE_DIR@
IBRD=@CMAKE_BINARY_DIR@

export PVKERNEL_PLUGIN_PATH=$IBRD/libpvkernel/plugins
export SQUEY_PLUGIN_PATH=$IBRD/libsquey/plugins
export QUERY_BUILDER_PATH=$ISRD/libpvkernel/src/widgets/querybuilder
export COPYING_DIR=$ISRD/COPYING
export PVFORMAT_HELPER=$ISRD/libpvkernel/plugins

# AG: we don't need this anymore, because
# the locale is automatically found for times in log files
# Moreover, it breaks Qt's qPrintable(QString) (because the wrong locale is choosen) and
# other displaying stuff that relies on the system's locale
# See also SVN commit #3081
#LANG=C
#export LC_ALL=C

#export SQUEY_LOG_FILE="log.txt"

VALGRIND_ALLOC_FNS="--alloc-fn=scalable_aligned_malloc --alloc-fn=scalable_malloc --alloc-fn=scalable_posix_memalign"

CMD_ARGS=("$@")

if [ "$1" == "debug" ]; then
	LOAD_PROJECT=""
	if [ "${2: -3}" == ".pv" ]; then
		LOAD_PROJECT="--project "
	fi
	export SQUEY_DEBUG_LEVEL="DEBUG"
	#export SQUEY_DEBUG_FILE="debug.txt"
	unset CMD_ARGS[0]
	gdb -ex run --args $IBRD/gui-qt/src/squey $LOAD_PROJECT ${CMD_ARGS[@]}
	exit 0
fi

if [ "$1" == "qdebug" ]; then
	qtcreator -debug $IBRD/gui-qt/src/squey
	exit 0
fi

if [ "$1" == "ddd" ]
then
export SQUEY_DEBUG_LEVEL="DEBUG"
#export SQUEY_DEBUG_FILE="debug.txt"
	ddd $IBRD/gui-qt/src/squey
	exit 0
fi
if [ "$1" == "nem" ]
then
#export SQUEY_DEBUG_LEVEL="DEBUG"
#export SQUEY_DEBUG_FILE="debug.txt"
	nemiver $IBRD/gui-qt/src/squey
	exit 0
fi
if [ "$1" == "debug-nogl" ]
then
export SQUEY_DEBUG_LEVEL="DEBUG"
#export SQUEY_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $IBRD/gui-qt/src/squey ${ARGS[@]}
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export SQUEY_DEBUG_LEVEL="NOTICE"
#export SQUEY_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $IBRD/gui-qt/src/squey ${ARGS[@]}
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --log-file=./valgrind.out --leak-check=full --track-origins=yes --show-reachable=yes $IBRD/gui-qt/src/squey
	exit 0
fi

if [ "$1" == "massif" ]
then
	valgrind $VALGRIND_ALLOC_FNS --depth=60 --tool=massif --heap=yes --detailed-freq=1 --threshold=0.1 $IBRD/gui-qt/src/squey
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export SQUEY_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no gui-qt/src/squey
	valgrind --tool=callgrind $IBRD/gui-qt/src/squey
	exit 0
fi

if [ "$1" == "calltrace" ]
then
	LD_PRELOAD=`pwd`/calltrace.so $IBRD/gui-qt/src/squey
	exit 0
fi

if [ "$1" == "gldebug" ]
then
	/usr/local/bin/gldb-gui gui-qt/src/squey
	exit 0
fi

if [ "$1" == "test" ]
then
#export SQUEY_DEBUG_FILE="debug.txt"
	gui-qt/src/squey test_petit.log
	exit 0
fi

export PATH=$PATH:$IBRD/libpvguiqt/src/
$IBRD/gui-qt/src/squey $LOAD_PROJECT $@
