/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldIPv4IPv6FromGUID.h"
#include "PVFieldSplitterIPv4IPv6FromGUIDParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("ipv4-ipv6_from_GUID", PVFilter::PVFieldIPv4IPv6FromGUID);
	REGISTER_CLASS_AS("splitter_ipv4-ipv6_from_GUID", PVFilter::PVFieldIPv4IPv6FromGUID, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("ipv4-ipv6_from_GUID", PVFilter::PVFieldSplitterIPv4IPv6FromGUIDParamWidget);
}
