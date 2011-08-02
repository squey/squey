// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVLayerFilterHeatline.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	//REGISTER_FILTER("Heatline/Colorize", Picviz::PVLayerFilterHeatlineColor);
	//REGISTER_FILTER("Heatline/Select", Picviz::PVLayerFilterHeatlineSel);
	REGISTER_FILTER("Frequency gradient", Picviz::PVLayerFilterHeatlineSelAndCol);
}
