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

PVCORE_PATH=$ISRC/libpvcore
PVRUSH_PATH=$ISRC/libpvrush
PVGL_DIR=$ISRC/libpvgl
PICVIZ_PATH=$ISRD/libpicviz

#export PICVIZ_LOG_FILE="log.txt"
export PVGL_SHARE_DIR=$ISRD/libpvgl/data/
export PICVIZ_DEBUG_LEVEL="INFO"
export PICVIZ_FILTERS_DIR=$PICVIZ_PATH/plugins/filters/
export PICVIZ_LAYER_FILTERS_DIR=$PICVIZ_PATH/plugins/layer-filters/
export PICVIZ_MAPPING_FILTERS_DIR=$PICVIZ_PATH/plugins/mapping-filters/
export PICVIZ_PLOTTING_FILTERS_DIR=$PICVIZ_PATH/plugins/plotting-filters/
export PICVIZ_FUNCTIONS_DIR=$PICVIZ_PATH/plugins/functions
# export PVRUSH_NORMALIZE_DIR=$PVRUSH_PATH/plugins/normalize 
# export PVRUSH_NORMALIZE_HELPERS_DIR=$PVRUSH_PATH/plugins/normalize-helpers 

export PVRUSH_NORMALIZE_DIR=libpvrush/plugins/normalize
export PVRUSH_NORMALIZE_HELPERS_DIR="libpvrush/plugins/normalize-helpers;/home/stricaud/pvrush-formats-extra/"

export PVRUSH_INPUTTYPE_DIR=libpvrush/plugins/input_types
export PVRUSH_SOURCE_DIR=libpvrush/plugins/sources

export PVFILTER_NORMALIZE_DIR=libpvfilter/plugins/normalize

# export PICVIZ_NORMALIZE_DIR=$PICVIZ_PATH/plugins/normalize 
# export PICVIZ_NORMALIZE_HELPERS_DIR=$PICVIZ_PATH/plugins/normalize-helpers 
export LD_LIBRARY_PATH=$PICVIZ_PATH/src/:$PVGL_PATH/src

if [ "$1" == "debug" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	gdb picviz-inspector/src/picviz-inspector
	exit 0
fi
if [ "$1" == "debug-nogl" ]
then
export PICVIZ_DEBUG_LEVEL="DEBUG"
#export PICVIZ_DEBUG_FILE="debug.txt"
	gdb picviz-inspector/src/picviz-inspector|egrep -v "PVGL"
#-e ".*PVGL.*" --invert-match
	exit 0
fi

if [ "$1" == "debug-quiet" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
#export PICVIZ_DEBUG_FILE="debug.txt"
	gdb picviz-inspector/src/picviz-inspector
	exit 0
fi

if [ "$1" == "valgrind" ]
then
	valgrind --leak-check=full --track-origins=yes picviz-inspector/src/picviz-inspector
	exit 0
fi

if [ "$1" == "valgrind-leaks" ]
then
	valgrind --db-attach=no --log-file=./valgrind.out --leak-check=yes picviz-inspector/src/picviz-inspector
	exit 0
fi

if [ "$1" == "callgrind" ]
then
export PICVIZ_DEBUG_LEVEL="NOTICE"
	#valgrind --tool=callgrind --instr-atstart=no picviz-inspector/src/picviz-inspector
	valgrind --tool=callgrind picviz-inspector/src/picviz-inspector
	exit 0
fi


if [ "$1" == "gldebug" ]
then
	/usr/local/bin/gldb-gui picviz-inspector/src/picviz-inspector
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
	#/usr/local/cuda/bin/cuda-memcheck --continue picviz-inspector/src/picviz-inspector >/tmp/cuda-test-log.txt 2>&1
	/usr/local/cuda/bin/cuda-gdb --quiet --nw picviz-inspector/src/picviz-inspector 
	#picviz-inspector/src/picviz-inspector
	#gedit /tmp/cuda-test-log.txt
	exit 0
fi

picviz-inspector/src/picviz-inspector



