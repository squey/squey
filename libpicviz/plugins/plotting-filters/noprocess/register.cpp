/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterNoprocess.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plotting types that do not process anyhting are registered here
	REGISTER_CLASS("enum_default", Picviz::PVPlottingFilterNoprocess);
	REGISTER_CLASS("host_default", Picviz::PVPlottingFilterNoprocess);
	REGISTER_CLASS("user-defined_default", Picviz::PVPlottingFilterNoprocess);
	REGISTER_CLASS("ipv4_default", Picviz::PVPlottingFilterNoprocess);
}
