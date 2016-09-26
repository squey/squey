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
#include "PVFieldSplitterURL.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("url", PVFilter::PVFieldSplitterURL);
	REGISTER_CLASS_AS("splitter_url", PVFilter::PVFieldSplitterURL, PVFilter::PVFieldsFilterReg);
}
