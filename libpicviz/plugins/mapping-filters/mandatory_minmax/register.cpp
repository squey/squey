/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMandatoryMappingFilterMinmax.h"

// This method will be called by libpicviz
// This mapping filter will be registered as a mandatory mapping filter ! (FilterT is set to PVMandatoryMappingFilter in PVMandatoryMappingFilter.h)
LibCPPExport void register_class()
{
	REGISTER_CLASS("mandatory_minmax", Picviz::PVMandatoryMappingFilterMinmax);
}
