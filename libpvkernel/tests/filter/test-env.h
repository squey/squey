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
	setenv("PVKERNEL_PLUGIN_PATH", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins/", 0);
}
