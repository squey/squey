// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVFieldSplitterURL.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("url", PVFilter::PVFieldSplitterURL);
	REGISTER_FILTER_AS("splitter_url", PVFilter::PVFieldSplitterURL, PVFilter::PVFieldsFilterReg);
}
