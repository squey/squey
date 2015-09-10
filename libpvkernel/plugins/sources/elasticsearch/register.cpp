/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include <pvkernel/core/PVClassLibrary.h>
#include "PVSourceCreatorElasticsearch.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("elasticsearch", PVRush::PVSourceCreatorElasticsearch);
}
