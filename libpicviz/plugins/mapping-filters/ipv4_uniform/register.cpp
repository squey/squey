/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

// Register the plugin in PVFilterLibrary
//

#include "PVMappingFilterIPv4Uniform.h"
#include <pvkernel/core/PVClassLibrary.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("ipv4_uniform", Picviz::PVMappingFilterIPv4Uniform);
}
