/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>

#include "PVSourceCreatorSplunk.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("splunk", PVRush::PVSourceCreatorSplunk);
}
