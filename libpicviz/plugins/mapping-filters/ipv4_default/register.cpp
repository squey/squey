/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterIPv4Default.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("ipv4_default", Picviz::PVMappingFilterIPv4Default);
}
