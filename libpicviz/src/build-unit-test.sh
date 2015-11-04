#!/bin/bash
#
# @file
#
# @copyright (C) Picviz Labs 2010-March 2015
# @copyright (C) ESI Group INENDI April 2015-2015

gcc -D_UNIT_TEST_ -ggdb filter-unit.c -o filter-unit -Iinclude/ `pkg-config apr-1 --cflags` `pkg-config apr-1 --libs`
gcc -D_UNIT_TEST_ -ggdb type-discovery.c -o type-discovery -Iinclude/ `pkg-config apr-1 --cflags` `pkg-config apr-1 --libs` -lpcre

