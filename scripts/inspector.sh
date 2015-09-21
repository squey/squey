#!/bin/bash

# \file inspector.sh
#
# Copyright (C) Picviz Labs 2010-2012

INSPECTOR_SOURCE_ROOT_DIR=$(/usr/bin/realpath $(dirname $0))
ISRD=$INSPECTOR_SOURCE_ROOT_DIR

PV_NOFILE=`ulimit -Hn`
ulimit -Sn "$PV_NOFILE"

cd $ISRD

$ISRD/picviz-inspector
