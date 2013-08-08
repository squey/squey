/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterIP.h"
#include "PVFieldSplitterIPParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("ip", PVFilter::PVFieldSplitterIP);
	REGISTER_CLASS_AS("splitter_ip", PVFilter::PVFieldSplitterIP, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("ip", PVFilter::PVFieldSplitterIPParamWidget);
}
