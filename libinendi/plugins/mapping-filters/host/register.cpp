/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterHost.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("host", Inendi::PVMappingFilterHost);
}
