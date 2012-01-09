#!/bin/bash

INSPECTOR_SOURCE_ROOT_DIR=.
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

# AG: we don't need this anymore, because
# the locale is automatically found for times in log files
# Moreover, it breaks Qt's qPrintable(QString) (because the wrong locale is choosen) and
# other displaying stuff that relies on the system's locale
# See also SVN commit #3081
#LANG=C
#export LC_ALL=C

PVKERNEL_PATH=$ISRC/libpvkernel
PVGL_DIR=$ISRC/libpvgl
PICVIZ_PATH=$ISRD/libpicviz

#export PICVIZ_LOG_FILE="log.txt"
export PVGL_SHARE_DIR=./libpvgl/data/
export PICVIZ_DEBUG_LEVEL="INFO"
export PICVIZ_LAYER_FILTERS_DIR=$PICVIZ_PATH/plugins/layer-filters/
export PICVIZ_MAPPING_FILTERS_DIR=$PICVIZ_PATH/plugins/mapping-filters/
export PICVIZ_PLOTTING_FILTERS_DIR=$PICVIZ_PATH/plugins/plotting-filters/
export PICVIZ_AXIS_COMPUTATION_PLUGINS_DIR=$PICVIZ_PATH/plugins/axis-computation/

export PVRUSH_NORMALIZE_HELPERS_DIR="libpvkernel/plugins/normalize-helpers;~/.pvrush-formats-extra"

export PVRUSH_INPUTTYPE_DIR=libpvkernel/plugins/input_types
export PVRUSH_SOURCE_DIR=libpvkernel/plugins/sources

export PVFILTER_NORMALIZE_DIR=libpvkernel/plugins/normalize

export LD_LIBRARY_PATH=$PVKERNEL_PATH/src/:$PICVIZ_PATH/src/:$PVGL_PATH/src

VALGRIND_ALLOC_FNS="--alloc-fn=scalable_aligned_malloc --alloc-fn=scalable_malloc --alloc-fn=scalable_posix_memalign"

CMD_ARGS=("$@")

if [ "$1" == "debug" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args gui-qt/src/picviz-inspector ${ARGS[@]}
	exit 0
fi
if [ "$1" == "ddd" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	ddd gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "debug-nogl" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args gui-qt/src/picviz-inspector ${ARGS[@]} |egrep -v "PVGL"
#-e ".*PVGL.*" --invert-match
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
#export PICVIZ_DEBUG_FILE="debug.txt"
	unset ARGS[0]
	gdb --args gui-qt/src/picviz-inspector ${ARGS[@]}
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --log-file=./valgrind.out --leak-check=full --track-origins=yes --show-reachable=yes gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "massif" ]
then
	valgrind $VALGRIND_ALLOC_FNS --depth=60 --tool=massif --heap=yes --detailed-freq=1 --threshold=0.1 gui-qt/src/picviz-inspector
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no gui-qt/src/picviz-inspector
	valgrind --tool=callgrind gui-qt/src/picviz-inspector
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

gui-qt/src/picviz-inspector $@
