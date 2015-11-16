/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVAxisComputationPython.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("python", Inendi::PVAxisComputationPython);
}
