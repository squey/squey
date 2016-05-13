/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>
#include "PVLayerFilterHeatline.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("frequency-gradient", Inendi::PVLayerFilterHeatline);
	DECLARE_TAG(PVAXIS_TAG_KEY, PVAXIS_TAG_KEY_DESC, Inendi::PVLayerFilterHeatline);
}
