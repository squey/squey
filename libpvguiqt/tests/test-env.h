/**
 * \file test-env.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdlib.h>
#include <pvbase/general.h>

#ifdef WIN32
#define pv_setenv(a,b,c) putenv(a "=" b)
#else
#define pv_setenv(a,b,c) setenv(a,b,c)
#endif

void init_env()
{
	pv_setenv("PVRUSH_NORMALIZE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/normalize",0);
	pv_setenv("PVFILTER_NORMALIZE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/normalize",0);
	pv_setenv("PVRUSH_NORMALIZE_HELPERS_DIR",PICVIZ_SOURCE_DIRECTORY "/libpvkernel/plugins/normalize-helpers:./test-formats",0);
	pv_setenv("PICVIZ_CACHE_DIR","./cache",0);
	pv_setenv("PVRUSH_INPUTTYPE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/input_types",0);
	pv_setenv("PVRUSH_SOURCE_DIR",PICVIZ_BUILD_DIRECTORY "/libpvkernel/plugins/sources",0);
	pv_setenv("PICVIZ_MAPPING_FILTERS_DIR",PICVIZ_BUILD_DIRECTORY "/libpicviz/plugins/mapping-filters",0);
	pv_setenv("PICVIZ_PLOTTING_FILTERS_DIR",PICVIZ_BUILD_DIRECTORY "/libpicviz/plugins/plotting-filters",0);
	pv_setenv("PICVIZ_LAYER_FILTERS_DIR",PICVIZ_BUILD_DIRECTORY "/libpicviz/plugins/layer-filters",0);
	pv_setenv("PICVIZ_ROW_FILTERS_DIR",PICVIZ_BUILD_DIRECTORY "/libpicviz/plugins/row-filters",0);
	pv_setenv("PVGL_SHARE_DIR",PICVIZ_SOURCE_DIRECTORY "/libpvgl/data/",0);
}
