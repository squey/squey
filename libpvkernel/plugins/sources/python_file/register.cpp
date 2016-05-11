/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorPythonfile.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("python_file", PVRush::PVSourceCreatorPythonfile);
}
