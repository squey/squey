#!/bin/bash


echo "Setting testing environnement for PVRush JNI..."

PICVIZ_PATH=../../../../..


export PVRUSH_INPUTTYPE_DIR=$PICVIZ_PATH/libpvrush/plugins/input_types
export PVRUSH_SOURCE_DIR=$PICVIZ_PATH/libpvrush/plugins/sources
export PVFILTER_NORMALIZE_DIR=$PICVIZ_PATH/libpvfilter/plugins/normalize
