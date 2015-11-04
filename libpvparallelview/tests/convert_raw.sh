#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2012-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

convert -size 1024x1024 -depth 8 rgba:$1 $1.png
