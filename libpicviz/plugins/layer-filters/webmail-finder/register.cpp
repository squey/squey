/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include "PVLayerFilterWebmailFinder.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("Find/Webmails", Picviz::PVLayerFilterWebmailFinder);
	DECLARE_TAG(PVAXIS_TAG_DOMAIN, PVAXIS_TAG_DOMAIN_DESC, Picviz::PVLayerFilterWebmailFinder);
}
