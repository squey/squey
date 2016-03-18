#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

INSPECTOR_SOURCE_ROOT_DIR=@CMAKE_BINARY_DIR@
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

export PVKERNEL_PLUGIN_PATH=$ISRD/libpvkernel/plugins
export INENDI_PLUGIN_PATH=$ISRD/libinendi/plugins

# Migration from picviz to inendi
if [ ! -d "$HOME/.inendi" ] && [ -d "$HOME/.picviz" ]
then
	mv "$HOME/.picviz" "$HOME/.inendi"
	ln -s "$HOME/.inendi" "$HOME/.picviz"
fi

if [ ! -d "$HOME/.config/ESI Group" ] && [ -d "$HOME/.config/Picviz Labs" ]
then
	mv "$HOME/.config/Picviz Labs" "$HOME/.config/ESI Group"
	mv "$HOME/.config/ESI Group/Picviz Inspector.conf" "$HOME/.config/ESI Group/INENDI Inspector.conf"
	ln -s "$HOME/.config/ESI Group" "$HOME/.config/Picviz Labs"
	ln -s "$HOME/.config/ESI Group/INENDI Inspector.conf" "$HOME/.config/ESI Group/Picviz Inspector.conf"
fi

if [ -d "$HOME/.config/Picviz" ] && [ ! -h "$HOME/.config/Picviz" ]
then
	mv "$HOME"/.config/Picviz/* "$HOME/.config/ESI Group"
	rmdir "$HOME/.config/Picviz"
	ln -s "$HOME/.config/ESI Group" "$HOME/.config/Picviz"
fi

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
	gdb -ex run --args $ISRD/gui-qt/src/inendi-inspector $LOAD_PROJECT ${CMD_ARGS[@]}
	exit 0
fi

if [ "$1" == "qdebug" ]; then
	qtcreator -debug $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "ddd" ]
then
export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	ddd $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi
if [ "$1" == "nem" ]
then
#export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	nemiver $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi
if [ "$1" == "debug-nogl" ]
then
export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $ISRD/gui-qt/src/inendi-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export INENDI_DEBUG_LEVEL="NOTICE"
#export INENDI_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $ISRD/gui-qt/src/inendi-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --log-file=./valgrind.out --leak-check=full --track-origins=yes --show-reachable=yes $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "massif" ]
then
	valgrind $VALGRIND_ALLOC_FNS --depth=60 --tool=massif --heap=yes --detailed-freq=1 --threshold=0.1 $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export INENDI_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no gui-qt/src/inendi-inspector
	valgrind --tool=callgrind $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "calltrace" ]
then
	LD_PRELOAD=`pwd`/calltrace.so $ISRD/gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "gldebug" ]
then
	/usr/local/bin/gldb-gui gui-qt/src/inendi-inspector
	exit 0
fi

if [ "$1" == "cuda-debug" ]
then
export INENDI_DEBUG_LEVEL="DEBUG"
#export INENDI_DEBUG_FILE="debug.txt"
cd libinendi/src/
make
cd ../..
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	#/usr/local/cuda/bin/cuda-memcheck --continue gui-qt/src/inendi-inspector >/tmp/cuda-test-log.txt 2>&1
	/usr/local/cuda/bin/cuda-gdb --quiet --nw gui-qt/src/inendi-inspector 
	#gui-qt/src/inendi-inspector
	#gedit /tmp/cuda-test-log.txt
	exit 0
fi

if [ "$1" == "test" ]
then
#export INENDI_DEBUG_FILE="debug.txt"
	gui-qt/src/inendi-inspector test_petit.log
	exit 0
fi
	
catchsegv $ISRD/gui-qt/src/inendi-inspector $LOAD_PROJECT $@
