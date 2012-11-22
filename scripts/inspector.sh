#!/bin/bash

# \file inspector.sh
#
# Copyright (C) Picviz Labs 2010-2012

INSPECTOR_SOURCE_ROOT_DIR=$(/usr/bin/realpath $(dirname $0))
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

cd $ISRD

PVCORE_PATH=$ISRD/
PVRUSH_PATH=$ISRD/
PVGL_DIR=$ISRD/
PICVIZ_PATH=$ISRD/

export PVGL_SHARE_DIR=$ISRD/share/
export PICVIZ_DEBUG_LEVEL="INFO"
export PICVIZ_LAYER_FILTERS_DIR=$ISRD/layer-filters/
export PICVIZ_MAPPING_FILTERS_DIR=$ISRD/mapping-filters/
export PICVIZ_PLOTTING_FILTERS_DIR=$ISRD/plotting-filters/
export PICVIZ_ROW_FILTERS_DIR=$ISRD/row-filters/
export PVRUSH_NORMALIZE_HELPERS_DIR=$ISRD/normalize-helpers/

export PVRUSH_INPUTTYPE_DIR=$ISRD/input-types/
export PVRUSH_SOURCE_DIR=$ISRD/sources/

export PVFILTER_NORMALIZE_DIR=$ISRD/normalize-filters/

export LD_LIBRARY_PATH=$ISRD/.:$ISRD/libtulip/lib:$LD_LIBRARY_PATH

$ISRD/picviz-inspector
