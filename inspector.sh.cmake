#!/bin/bash
#
# @file
#

ISRD=@CMAKE_SOURCE_DIR@
IBRD=@CMAKE_BINARY_DIR@

export PVKERNEL_PLUGIN_PATH=$IBRD/libpvkernel/plugins
export INENDI_PLUGIN_PATH=$IBRD/libinendi/plugins
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

#export INENDI_LOG_FILE="log.txt"

VALGRIND_ALLOC_FNS="--alloc-fn=scalable_aligned_malloc --alloc-fn=scalable_malloc --alloc-fn=scalable_posix_memalign"

CMD_ARGS=("$@")

if [ "$1" == "debug" ]; then
	LOAD_PROJECT=""
	if [ "${2: -3}" == ".pv" ]; then
		LOAD_PROJECT="--project "
	fi
	export INENDI_DEBUG_LEVEL="DEBUG"
	#export INENDI_DEBUG_FILE="debug.txt"
	unset CMD_ARGS[0]
	gdb -ex run --args $IBRD/gui-qt/src/inendi-inspector $LOAD_PROJECT ${CMD_ARGS[@]}
	exit 0
fi

if [ "$1" == "qdebug" ]; then
	qtcreator -debug $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "ddd" ]
then
export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	ddd $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi
if [ "$1" == "nem" ]
then
#export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	nemiver $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi
if [ "$1" == "debug-nogl" ]
then
export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $IBRD/gui-qt/src/inendi-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export INENDI_DEBUG_LEVEL="NOTICE"
#export INENDI_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $IBRD/gui-qt/src/inendi-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --log-file=./valgrind.out --leak-check=full --track-origins=yes --show-reachable=yes $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "massif" ]
then
	valgrind $VALGRIND_ALLOC_FNS --depth=60 --tool=massif --heap=yes --detailed-freq=1 --threshold=0.1 $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export INENDI_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no gui-qt/src/inendi-inspector
	valgrind --tool=callgrind $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "calltrace" ]
then
	LD_PRELOAD=`pwd`/calltrace.so $IBRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "gldebug" ]
then
	/usr/local/bin/gldb-gui gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "test" ]
then
#export INENDI_DEBUG_FILE="debug.txt"
	gui-qt/src/inendi-inspector test_petit.log
	exit 0
fi

export PATH=$PATH:$IBRD/libpvguiqt/src/
$IBRD/gui-qt/src/inendi-inspector $LOAD_PROJECT $@
