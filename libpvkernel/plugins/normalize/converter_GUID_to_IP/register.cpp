/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2013
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldGUIDToIP.h"
#include "PVFieldConverterGUIDToIPParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("GUID_to_IP", PVFilter::PVFieldGUIDToIP);
	REGISTER_CLASS_AS("converter_GUID_to_IP", PVFilter::PVFieldGUIDToIP, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("GUID_to_IP", PVFilter::PVFieldConverterGUIDToIPParamWidget);
}
