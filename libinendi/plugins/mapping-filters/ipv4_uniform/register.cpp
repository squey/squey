/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

// Register the plugin in PVFilterLibrary
//

#include "PVMappingFilterIPv4Uniform.h"
#include <pvkernel/core/PVClassLibrary.h>

// This method will be called by libinendi
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("ipv4_uniform", Inendi::PVMappingFilterIPv4Uniform);
}
