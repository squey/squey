/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVRFFAxesBindNearestNeighbors.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("axes-bind-nearest-neighbors", Picviz::PVRFFAxesBindNearestNeighbors);
}
