#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

export LD_LIBRARY_PATH=$(/usr/bin/realpath $(dirname $0)):$LD_LIBRARY_PATH
export PATH=$(/usr/bin/realpath $(dirname $0)):$PATH

inendi-inspector
