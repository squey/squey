/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterMacAddress.h"
#include "PVFieldSplitterMacAddressParamWidget.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("mac_address", PVFilter::PVFieldSplitterMacAddress);
	REGISTER_CLASS_AS("splitter_mac_address", PVFilter::PVFieldSplitterMacAddress, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("mac_address", PVFilter::PVFieldSplitterMacAddressParamWidget);
}
