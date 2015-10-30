/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
