/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterString4Bsort.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("string_4Bsort", Picviz::PVMappingFilterString4Bsort);
}
