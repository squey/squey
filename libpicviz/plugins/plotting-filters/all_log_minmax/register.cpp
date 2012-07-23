/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterLogMinmax.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plottign types that can plot vs. their min and max values are registered here
	REGISTER_CLASS("float_log", Picviz::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("integer_log", Picviz::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("string_log", Picviz::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("time_log", Picviz::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("ipv4_log", Picviz::PVPlottingFilterLogMinmax);
}
