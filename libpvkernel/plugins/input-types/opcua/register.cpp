/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

// Register the plugin in PVFilterLibrary
//
#include <pvbase/general.h>
#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeOpcUa.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	REGISTER_CLASS("opcua", PVRush::PVInputTypeOpcUa);
}