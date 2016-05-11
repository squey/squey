/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterIP.h"
#include "PVFieldSplitterIPParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("ip", PVFilter::PVFieldSplitterIP);
	REGISTER_CLASS_AS("splitter_ip", PVFilter::PVFieldSplitterIP, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("ip", PVFilter::PVFieldSplitterIPParamWidget);
}
