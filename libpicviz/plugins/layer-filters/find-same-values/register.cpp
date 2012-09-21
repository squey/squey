/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include "PVLayerFilterFindSameValues.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("find-same-values", Picviz::PVLayerFilterFindSameValues);
	DECLARE_TAG(PVAXIS_TAG_KEY, PVAXIS_TAG_KEY_DESC, Picviz::PVLayerFilterFindSameValues);
}
