/**
 * @file
 *
 * @copyright (C) Picviz Labs 2015
*/

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeSplunk.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("splunk", PVRush::PVInputTypeSplunk);
}
