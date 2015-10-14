/**
 * \file test-env.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <cstdlib>

void init_env()
{
	setenv("PVFILTER_NORMALIZE_DIR","../../libpvkernel/plugins/normalize",0);
	setenv("PVRUSH_NORMALIZE_HELPERS_DIR","../../libpvkernel/plugins/normalize-helpers:./test-formats",0);
	setenv("PICVIZ_CACHE_DIR","./cache",0);
	setenv("PVRUSH_INPUTTYPE_DIR","../../libpvkernel/plugins/input_types",0);
	setenv("PVRUSH_SOURCE_DIR","../../libpvkernel/plugins/sources",0);
	setenv("PICVIZ_MAPPING_FILTERS_DIR","../../libpicviz/plugins/mapping-filters",0);
	setenv("PICVIZ_PLOTTING_FILTERS_DIR","../../libpicviz/plugins/plotting-filters",0);
}
