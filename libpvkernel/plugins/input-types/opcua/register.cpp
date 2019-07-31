/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

// Register the plugin in PVFilterLibrary
//
#include <License.h>
#include <pvbase/general.h>
#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeOpcUa.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	if (Inendi::Utils::License::RAII_LicenseFeature::is_available(INENDI_LICENSE_PREFIX,
	                                                              "INSPECTOR_MODULE_ES")) {
		REGISTER_CLASS("opcua", PVRush::PVInputTypeOpcUa);
	}
}