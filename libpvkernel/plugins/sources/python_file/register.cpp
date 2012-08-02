/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorPythonfile.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("python_file", PVRush::PVSourceCreatorPythonfile);
}
