/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterRegexp.h"
#include "PVFieldSplitterRegexpParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("regexp", PVFilter::PVFieldSplitterRegexp);
	REGISTER_CLASS_AS("splitter_regexp", PVFilter::PVFieldSplitterRegexp, PVFilter::PVFieldsFilterReg);
	
	REGISTER_CLASS("regexp", PVFilter::PVFieldSplitterRegexpParamWidget);
}
