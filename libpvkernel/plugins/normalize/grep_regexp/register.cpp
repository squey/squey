/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldFilterRegexpGrep.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("regexp", PVFilter::PVFieldFilterRegexpGrep);
	REGISTER_CLASS_AS("filter_regexp", PVFilter::PVFieldFilterRegexpGrep, PVFilter::PVFieldsFilterReg);
}
