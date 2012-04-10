// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>
#include "PVLayerFilterHeatline.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	//REGISTER_CLASS("Heatline/Colorize", Picviz::PVLayerFilterHeatlineColor);
	//REGISTER_CLASS("Heatline/Select", Picviz::PVLayerFilterHeatlineSel);
	REGISTER_CLASS("frequency-gradient", Picviz::PVLayerFilterHeatlineSelAndCol);
	DECLARE_TAG(PVAXIS_TAG_KEY, PVAXIS_TAG_KEY_DESC, Picviz::PVLayerFilterHeatlineSelAndCol);
}
