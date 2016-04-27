/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterMinmax.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	// All the plottign types that can plot vs. their min and max values are registered here
	REGISTER_CLASS("float_default", Inendi::PVPlottingFilterMinmax);
	REGISTER_CLASS("integer_default", Inendi::PVPlottingFilterMinmax);
	REGISTER_CLASS("string_default", Inendi::PVPlottingFilterMinmax);
	REGISTER_CLASS("time_minmax", Inendi::PVPlottingFilterMinmax);
	REGISTER_CLASS("ipv4_minmax", Inendi::PVPlottingFilterMinmax);
	REGISTER_CLASS("date_default", Inendi::PVPlottingFilterMinmax);
}
