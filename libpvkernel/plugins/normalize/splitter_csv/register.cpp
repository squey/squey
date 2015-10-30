/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterCSV.h"
#include "PVFieldSplitterCSVParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	// Register as a field splitter and as a general fields filter
	REGISTER_CLASS("csv", PVFilter::PVFieldSplitterCSV);
	REGISTER_CLASS_AS("splitter_csv", PVFilter::PVFieldSplitterCSV, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("csv", PVFilter::PVFieldSplitterCSVParamWidget);
}
