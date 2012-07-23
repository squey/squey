#!/bin/bash

# \file iceccize.sh
#
# Copyright (C) Picviz Labs 2010-2012

rm CMakeCache.txt
#make clean
CC=icecc CXX=icecc cmake .
