/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldFilterRegexpGrep.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("regexp", PVFilter::PVFieldFilterRegexpGrep);
	REGISTER_CLASS_AS("filter_regexp", PVFilter::PVFieldFilterRegexpGrep,
	                  PVFilter::PVFieldsFilterReg);
}
