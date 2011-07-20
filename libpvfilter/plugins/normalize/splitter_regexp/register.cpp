// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVFieldSplitterRegexp.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("regexp", PVFilter::PVFieldSplitterRegexp);
	REGISTER_FILTER_AS("splitter_regexp", PVFilter::PVFieldSplitterRegexp, PVFilter::PVFieldsFilterReg);
}
