/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include <pvbase/general.h>
#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeSplunk.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	REGISTER_CLASS("splunk", PVRush::PVInputTypeSplunk);
}
