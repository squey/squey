/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include "PVMappingFilterEnumDefault.h"
#include <pvkernel/core/PVClassLibrary.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("enum_default", Picviz::PVMappingFilterEnumDefault);
}
