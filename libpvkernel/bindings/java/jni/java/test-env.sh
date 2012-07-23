#!/bin/bash

# \file test-env.sh
#
# Copyright (C) Picviz Labs 2010-2012

echo "Setting testing environnement for PVRush JNI..."

PLUGINS=../../../../plugins/


export PVRUSH_INPUTTYPE_DIR=$PLUGINS/input_types
export PVRUSH_SOURCE_DIR=$PLUGINS/sources
export PVFILTER_NORMALIZE_DIR=$PLUGINS/normalize
