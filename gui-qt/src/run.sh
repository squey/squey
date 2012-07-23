#!/bin/bash

# \file run.sh
#
# Copyright (C) Picviz Labs 2010-2012

LIBPICVIZPATH=../../libpicviz

DEBUG_LEVEL_NORMAL="DEBUG"

if [ "$1" == "callgrind" ]
then
   PVGL_SHARE_DIR=$LIBPICVIZPATH/../libpvgl/data/ PICVIZ_FILTERS_DIR=$LIBPICVIZPATH/plugins/filters/  PICVIZ_FUNCTIONS_DIR=$LIBPICVIZPATH/plugins/functions PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ valgrind --tool=callgrind ./picviz-inspector
exit 0
fi

if [ "$1" == "valgrind" ]
then
   PVGL_SHARE_DIR=$LIBPICVIZPATH/../libpvgl/data/ PICVIZ_FILTERS_DIR=$LIBPICVIZPATH/plugins/filters/  PICVIZ_FUNCTIONS_DIR=$LIBPICVIZPATH/plugins/functions PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ valgrind --track-origins=yes --leak-check=full ./picviz-inspector
exit 0
fi

if [ "$1" == "log_debug" ]
then
	echo "Running with Loglevel : DEBUG"
	PVGL_SHARE_DIR=$LIBPICVIZPATH/../libpvgl/data/ PICVIZ_FILTERS_DIR=$LIBPICVIZPATH/plugins/filters/ PICVIZ_DEBUG_LEVEL="DEBUG" PICVIZ_FUNCTIONS_DIR=$LIBPICVIZPATH/plugins/functions PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ ./picviz-inspector
exit 0
fi


if [ "$1" == "debug" ]
then
	PVGL_SHARE_DIR=$LIBPICVIZPATH/../libpvgl/data/ LANG=C PICVIZ_FILTERS_DIR=$LIBPICVIZPATH/plugins/filters/ PICVIZ_DEBUG_LEVEL="NOTICE" PICVIZ_FUNCTIONS_DIR=$LIBPICVIZPATH/plugins/functions PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ gdb picviz-inspector
else
	PVGL_SHARE_DIR=$LIBPICVIZPATH/../libpvgl/data/ PICVIZ_FILTERS_DIR=$LIBPICVIZPATH/plugins/filters/ PICVIZ_DEBUG_LEVEL="$DEBUG_LEVEL_NORMAL" PICVIZ_FUNCTIONS_DIR=$LIBPICVIZPATH/plugins/functions PICVIZ_NORMALIZE_DIR=$LIBPICVIZPATH/plugins/normalize PICVIZ_NORMALIZE_HELPERS_DIR=$LIBPICVIZPATH/plugins/normalize-helpers LD_LIBRARY_PATH=$LIBPICVIZPATH/src/ ./picviz-inspector
fi
