/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldDuplicate.h"
#include "PVFieldSplitterDuplicateParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("duplicate", PVFilter::PVFieldDuplicate);
	REGISTER_CLASS_AS("splitter_duplicate", PVFilter::PVFieldDuplicate, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("duplicate", PVFilter::PVFieldSplitterDuplicateParamWidget);
}
