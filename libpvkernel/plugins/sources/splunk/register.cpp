/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/PVClassLibrary.h>

#include "PVSourceCreatorSplunk.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("splunk", PVRush::PVSourceCreatorSplunk);
}
