#!/bin/bash

# \file inspector.sh
#
# Copyright (C) Picviz Labs 2010-2012

INSPECTOR_SOURCE_ROOT_DIR=$(/usr/bin/realpath $(dirname $0))
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

PV_NOFILE=`ulimit -Hn`
ulimit -Sn "$PV_NOFILE"

cd $ISRD

PVCORE_PATH=$ISRD/
PVRUSH_PATH=$ISRD/
PICVIZ_PATH=$ISRD/

export PICVIZ_DEBUG_LEVEL="INFO"
export PICVIZ_QUERYBUILDER_DIR=$ISRD/querybuilder
export PVRUSH_NORMALIZE_HELPERS_DIR=$ISRD/normalize-helpers/

export PVRUSH_INPUTTYPE_DIR=$ISRD/input-types/
export PVRUSH_SOURCE_DIR=$ISRD/sources/

export PVFILTER_NORMALIZE_DIR=$ISRD/normalize-filters/

export LD_LIBRARY_PATH=$ISRD/.:$ISRD/libtulip/lib:$LD_LIBRARY_PATH

$ISRD/picviz-inspector
