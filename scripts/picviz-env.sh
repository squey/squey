#!/bin/bash

# \file picviz-env.sh
#
# Copyright (C) Picviz Labs 2010-2012

INSPECTOR_SOURCE_ROOT_DIR="/opt/picviz-inspector"
export PATH=$PATH:$INSPECTOR_SOURCE_ROOT_DIR
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

PVCORE_PATH=$ISRD/
PVRUSH_PATH=$ISRD/
PICVIZ_PATH=$ISRD/

export PICVIZ_DEBUG_LEVEL="INFO"
export PVRUSH_NORMALIZE_HELPERS_DIR=$ISRD/normalize-helpers/

export PVRUSH_INPUTTYPE_DIR=$ISRD/input-types/
export PVRUSH_SOURCE_DIR=$ISRD/sources/

export PVFILTER_NORMALIZE_DIR=$ISRD/normalize-filters/
