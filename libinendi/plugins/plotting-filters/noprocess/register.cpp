/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterNoprocess.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plotting types that do not process anyhting are registered here
	REGISTER_CLASS("enum_default", Inendi::PVPlottingFilterNoprocess);
	REGISTER_CLASS("host_default", Inendi::PVPlottingFilterNoprocess);
	REGISTER_CLASS("user-defined_default", Inendi::PVPlottingFilterNoprocess);
	REGISTER_CLASS("ipv4_default", Inendi::PVPlottingFilterNoprocess);
}
