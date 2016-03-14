/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVPLUGINS_H
#define PVRUSH_PVPLUGINS_H

#include <pvkernel/core/general.h>
#include <QString>

#define INPUT_TYPE_PREFIX "input_type"
#define SOURCE_PREFIX "source"

namespace PVRush {

namespace PVPluginsLoad
{
	int load_all_plugins();
	int load_input_type_plugins();
	int load_source_plugins();
	QString get_input_type_dir();
	QString get_source_dir();
};

}

#endif
