/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include <License.h>
#include <pvbase/general.h>
#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVInputType.h>

#include "PVInputTypeSplunk.h"

// This method will be called by libpvrush
LibCPPExport void register_class()
{
	if (Inendi::Utils::License::RAII_LicenseFeature::is_available(INENDI_LICENSE_PREFIX,
	                                                              "INSPECTOR_MODULE_SPLUNK")) {
		REGISTER_CLASS("splunk", PVRush::PVInputTypeSplunk);
	}
}
