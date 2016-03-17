/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLUGINS_H
#define PVFILTER_PVPLUGINS_H

#include <pvkernel/core/general.h>
#include <string>

#define NORMALIZE_FILTER_PREFIX "normalize"

namespace PVFilter {

namespace PVPluginsLoad
{
	int load_all_plugins();
	int load_normalize_plugins();
	std::string get_normalize_dir();
};

}

#endif
