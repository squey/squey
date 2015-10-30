/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVPLUGINS_H
#define PVFILTER_PVPLUGINS_H

#include <pvkernel/core/general.h>
#include <QString>

#define NORMALIZE_FILTER_PREFIX "normalize"

namespace PVFilter {

class PVPluginsLoad
{
public:
	static int load_all_plugins();
	static int load_normalize_plugins();
	static QString get_normalize_dir();
};

}

#endif
