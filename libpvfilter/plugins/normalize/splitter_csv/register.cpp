// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldSplitterCSV.h"
#include "PVFieldSplitterCSVParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("csv", PVFilter::PVFieldSplitterCSV);
	REGISTER_CLASS("csv", PVFilter::PVFieldSplitterCSVParamWidget);
}
