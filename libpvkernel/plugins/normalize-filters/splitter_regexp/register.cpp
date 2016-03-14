/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterRegexp.h"
#include "PVFieldSplitterRegexpParamWidget.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("regexp", PVFilter::PVFieldSplitterRegexp);
	REGISTER_CLASS_AS("splitter_regexp", PVFilter::PVFieldSplitterRegexp, PVFilter::PVFieldsFilterReg);
	
	REGISTER_CLASS("regexp", PVFilter::PVFieldSplitterRegexpParamWidget);
}
