#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

rm CMakeCache.txt
#make clean
CC=icecc CXX=icecc cmake .
