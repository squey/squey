/**
 * \file test-env.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvbase/general.h>

#include <cstdlib>

void init_env()
{
	setenv("PVFILTER_NORMALIZE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/normalize",0);
	setenv("PVRUSH_NORMALIZE_HELPERS_DIR",PICVIZ_SOURCE_DIRECTORY "/libpvkernel/plugins/normalize-helpers:./test-formats",0);
	setenv("PICVIZ_CACHE_DIR","./cache",0);
	setenv("PVRUSH_INPUTTYPE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/input_types",0);
	setenv("PVRUSH_SOURCE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/sources",0);
}
