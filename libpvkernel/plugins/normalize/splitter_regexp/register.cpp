// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldSplitterRegexp.h"
#include "PVFieldSplitterRegexpParamWidget.h"

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	REGISTER_FILTER("regexp", PVFilter::PVFieldSplitterRegexp);
	REGISTER_FILTER_AS("splitter_regexp", PVFilter::PVFieldSplitterRegexp, PVFilter::PVFieldsFilterReg);
	
	REGISTER_CLASS("regexp", PVFilter::PVFieldSplitterRegexpParamWidget);
}
