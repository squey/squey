/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterIpv4Default.h"

#include <QVariant>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("ipv4_default", Picviz::PVPlottingFilterIpv4Default);
}
