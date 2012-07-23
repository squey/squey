#!/bin/bash

# \file whatismissing.sh
#
# Copyright (C) Picviz Labs 2010-2012

git status |grep -v "tests/rush/Trush_" |grep -v "tests/core/Tcore_" |grep -v ".cmake" |grep -v "*~" |grep -v "*.cxx" |grep -v "*.so"
