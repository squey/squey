// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include <pvfilter/PVFilterFunction.h>
#include "PVFieldSplitterCSV.h"
#include "PVFieldSplitterCSVParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register as a field splitter and as a general fields filter
	REGISTER_FILTER("csv", PVFilter::PVFieldSplitterCSV);
	REGISTER_FILTER_AS("splitter_csv", PVFilter::PVFieldSplitterCSV, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("csv", PVFilter::PVFieldSplitterCSVParamWidget);
}
