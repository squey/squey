#!/bin/bash

# \file build-unit-test.sh
#
# Copyright (C) Picviz Labs 2010-2012

gcc -D_UNIT_TEST_ -ggdb filter-unit.c -o filter-unit -Iinclude/ `pkg-config apr-1 --cflags` `pkg-config apr-1 --libs`
gcc -D_UNIT_TEST_ -ggdb type-discovery.c -o type-discovery -Iinclude/ `pkg-config apr-1 --cflags` `pkg-config apr-1 --libs` -lpcre

