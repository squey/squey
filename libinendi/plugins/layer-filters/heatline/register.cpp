/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>
#include "PVLayerFilterHeatline.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	//REGISTER_CLASS("Heatline/Colorize", Inendi::PVLayerFilterHeatlineColor);
	//REGISTER_CLASS("Heatline/Select", Inendi::PVLayerFilterHeatlineSel);
	REGISTER_CLASS("frequency-gradient", Inendi::PVLayerFilterHeatlineSelAndCol);
	DECLARE_TAG(PVAXIS_TAG_KEY, PVAXIS_TAG_KEY_DESC, Inendi::PVLayerFilterHeatlineSelAndCol);
}