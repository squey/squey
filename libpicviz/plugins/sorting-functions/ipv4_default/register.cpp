/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVIPv4DefaultSortingFunc.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("ipv4_default", Picviz::PVIPv4DefaultSortingFunc);
}
