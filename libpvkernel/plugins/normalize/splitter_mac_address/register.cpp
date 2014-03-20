/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterMacAddress.h"
#include "PVFieldSplitterMacAddressParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("mac_address", PVFilter::PVFieldSplitterMacAddress);
	REGISTER_CLASS_AS("splitter_mac_address", PVFilter::PVFieldSplitterMacAddress, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("mac_address", PVFilter::PVFieldSplitterMacAddressParamWidget);
}
