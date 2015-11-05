/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterLogMinmax.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plottign types that can plot vs. their min and max values are registered here
	REGISTER_CLASS("float_log", Inendi::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("integer_log", Inendi::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("string_log", Inendi::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("time_log", Inendi::PVPlottingFilterLogMinmax);
	REGISTER_CLASS("ipv4_log", Inendi::PVPlottingFilterLogMinmax);
}
