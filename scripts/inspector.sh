#!/bin/bash

# \file inspector.sh
#
# Copyright (C) Picviz Labs 2010-2012
# Copyright (C) ESI Group INENDI 2015

export LD_LIBRARY_PATH=$(/usr/bin/realpath $(dirname $0)):$LD_LIBRARY_PATH
export PATH=$(/usr/bin/realpath $(dirname $0)):$PATH

picviz-inspector
