/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>

#include "PVSourceCreatorPcap.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("pcap", PVPcapsicum::PVSourceCreatorPcap);
}
