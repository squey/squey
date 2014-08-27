/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterKeyValue.h"
#include "PVFieldSplitterKeyValueParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("key_value", PVFilter::PVFieldSplitterKeyValue);
	REGISTER_CLASS_AS("splitter_key_value", PVFilter::PVFieldSplitterKeyValue, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("key_value", PVFilter::PVFieldSplitterKeyValueParamWidget);
}
