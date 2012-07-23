/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include "PVLayerFilterEntropy.h"
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>


// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("entropy", Picviz::PVLayerFilterEntropy);

	DECLARE_TAG(PVAXIS_TAG_USER_AGENT, PVAXIS_TAG_USER_AGENT_DESC, Picviz::PVLayerFilterEntropy);
	DECLARE_TAG(PVAXIS_TAG_SRCPORT, PVAXIS_TAG_SRCPORT_DESC, Picviz::PVLayerFilterEntropy);
	DECLARE_TAG(PVAXIS_TAG_DSTPORT, PVAXIS_TAG_DSTPORT_DESC, Picviz::PVLayerFilterEntropy);
	DECLARE_TAG(PVAXIS_TAG_SRCIP, PVAXIS_TAG_SRCIP_DESC, Picviz::PVLayerFilterEntropy);
	DECLARE_TAG(PVAXIS_TAG_DSTIP, PVAXIS_TAG_DSTIP_DESC, Picviz::PVLayerFilterEntropy);

}
