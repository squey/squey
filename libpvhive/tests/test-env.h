/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <cstdlib>

void init_env()
{
	setenv("PVFILTER_NORMALIZE_DIR", "../../libpvkernel/plugins/normalize", 0);
	setenv("PVRUSH_NORMALIZE_HELPERS_DIR",
	       "../../libpvkernel/plugins/normalize-helpers:./test-formats", 0);
	setenv("INENDI_CACHE_DIR", "./cache", 0);
	setenv("PVRUSH_INPUTTYPE_DIR", "../../libpvkernel/plugins/input_types", 0);
	setenv("PVRUSH_SOURCE_DIR", "../../libpvkernel/plugins/sources", 0);
	setenv("INENDI_MAPPING_FILTERS_DIR", "../../libinendi/plugins/mapping-filters", 0);
	setenv("INENDI_PLOTTING_FILTERS_DIR", "../../libinendi/plugins/plotting-filters", 0);
}
