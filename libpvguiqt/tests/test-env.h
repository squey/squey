/**
 * \file test-env.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <stdlib.h>


#ifdef WIN32
#define pv_setenv(a,b,c) putenv(a "=" b)
#else
#define pv_setenv(a,b,c) setenv(a,b,c)
#endif

void init_env()
{
	pv_setenv("PVRUSH_NORMALIZE_DIR","../../libpvkernel/plugins/normalize",0);
	pv_setenv("PVFILTER_NORMALIZE_DIR","../../libpvkernel/plugins/normalize",0);
	pv_setenv("PVRUSH_NORMALIZE_HELPERS_DIR","../../libpvkernel/plugins/normalize-helpers:./test-formats",0);
	pv_setenv("PICVIZ_CACHE_DIR","./cache",0);
	pv_setenv("PVRUSH_INPUTTYPE_DIR","../../libpvkernel/plugins/input_types",0);
	pv_setenv("PVRUSH_SOURCE_DIR","../../libpvkernel/plugins/sources",0);
	pv_setenv("PICVIZ_MAPPING_FILTERS_DIR","../../libpicviz/plugins/mapping-filters",0);
	pv_setenv("PICVIZ_PLOTTING_FILTERS_DIR","../../libpicviz/plugins/plotting-filters",0);
	pv_setenv("PICVIZ_ROW_FILTERS_DIR","../../libpicviz/plugins/row-filters",0);
	pv_setenv("PVGL_SHARE_DIR","../../libpvgl/data/",0);
}