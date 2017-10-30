/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "pcap/PVInputTypePcap.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	// Register under a unique name
	REGISTER_CLASS("pcap", PVPcapsicum::PVInputTypePcap);
}
