#!/bin/bash

# \file inspector.sh.cmake
#
# Copyright (C) Picviz Labs 2010-2012

INSPECTOR_SOURCE_ROOT_DIR=@CMAKE_BINARY_DIR@
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

# AG: we don't need this anymore, because
# the locale is automatically found for times in log files
# Moreover, it breaks Qt's qPrintable(QString) (because the wrong locale is choosen) and
# other displaying stuff that relies on the system's locale
# See also SVN commit #3081
#LANG=C
#export LC_ALL=C

PVKERNEL_PATH=$ISRD/libpvkernel
PICVIZ_PATH=$ISRD/libpicviz

#export PICVIZ_LOG_FILE="log.txt"
export PVGL_SHARE_DIR=./libpvgl/data/
if test -z "$PICVIZ_DEBUG_LEVEL"
then
	export PICVIZ_DEBUG_LEVEL="INFO"
fi
export PICVIZ_LAYER_FILTERS_DIR=$PICVIZ_PATH/plugins/layer-filters/
export PICVIZ_MAPPING_FILTERS_DIR=$PICVIZ_PATH/plugins/mapping-filters/
export PICVIZ_PLOTTING_FILTERS_DIR=$PICVIZ_PATH/plugins/plotting-filters/
export PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR=$PICVIZ_PATH/plugins/axis-computation/
export PICVIZ_SORTING_FUNCTIONS_PLUGINS_DIR=$PICVIZ_PATH/plugins/sorting-functions/
export PICVIZ_ROW_FILTERS_DIR=$PICVIZ_PATH/plugins/row-filters/

export PVRUSH_NORMALIZE_HELPERS_DIR="@CMAKE_SOURCE_DIR@/libpvkernel/plugins/normalize-helpers;~/.pvrush-formats-extra"

export PVRUSH_INPUTTYPE_DIR=$PVKERNEL_PATH/plugins/input_types
export PVRUSH_SOURCE_DIR=$PVKERNEL_PATH/plugins/sources

export PVFILTER_NORMALIZE_DIR=$PVKERNEL_PATH/plugins/normalize

VALGRIND_ALLOC_FNS="--alloc-fn=scalable_aligned_malloc --alloc-fn=scalable_malloc --alloc-fn=scalable_posix_memalign"

CMD_ARGS=("$@")

if [ "$1" == "debug" ]; then
	LOAD_PROJECT=""
	if [ "${2: -3}" == ".pv" ]; then
		LOAD_PROJECT="--project "
	fi
	export PICVIZ_DEBUG_LEVEL="DEBUG"
	#export PICVIZ_DEBUG_FILE="debug.txt"
	unset CMD_ARGS[0]
	gdb -ex run --args $ISRD/gui-qt/src/picviz-inspector $LOAD_PROJECT ${CMD_ARGS[@]}
	exit 0
fi

if [ "$1" == "qdebug" ]; then
	qtcreator -debug $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "ddd" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	ddd $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi
if [ "$1" == "nem" ]
then
#export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	nemiver $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi
if [ "$1" == "debug-nogl" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $ISRD/gui-qt/src/picviz-inspector ${ARGS[@]} |egrep -v "PVGL"
#-e ".*PVGL.*" --invert-match
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
#export PICVIZ_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args $ISRD/gui-qt/src/picviz-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --log-file=./valgrind.out --leak-check=full --track-origins=yes --show-reachable=yes $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "massif" ]
then
	valgrind $VALGRIND_ALLOC_FNS --depth=60 --tool=massif --heap=yes --detailed-freq=1 --threshold=0.1 $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no gui-qt/src/picviz-inspector
	valgrind --tool=callgrind $ISRD/gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "gldebug" ]
then
	/usr/local/bin/gldb-gui gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "cuda-debug" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
cd libpicviz/src/
make
cd ../..
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	#/usr/local/cuda/bin/cuda-memcheck --continue gui-qt/src/picviz-inspector >/tmp/cuda-test-log.txt 2>&1
	/usr/local/cuda/bin/cuda-gdb --quiet --nw gui-qt/src/picviz-inspector 
	#gui-qt/src/picviz-inspector
	#gedit /tmp/cuda-test-log.txt
	exit 0
fi

if [ "$1" == "test" ]
then
#export PICVIZ_DEBUG_FILE="debug.txt"
	gui-qt/src/picviz-inspector test_petit.log
	exit 0
fi
	
catchsegv $ISRD/gui-qt/src/picviz-inspector $LOAD_PROJECT $@
