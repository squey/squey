#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

git status |grep -v "tests/rush/Trush_" |grep -v "tests/core/Tcore_" |grep -v ".cmake" |grep -v "*~" |grep -v "*.cxx" |grep -v "*.so"
