/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterDateMonth.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("date_month", Picviz::PVMappingFilterDateMonth);
}
