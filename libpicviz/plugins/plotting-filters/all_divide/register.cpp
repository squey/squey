// Register the plugin in PVFilterLibrary
//

#include <pvkernel/filter/PVFilterLibrary.h>
#include "PVPlottingFilterAllDivide.h"

#include <QVariant>
#include <picviz/limits.h>

// This method will be called by libpicviz
LibCPPExport void register_filter()
{
	// Register under the name "type_format"
	PVCore::PVArgumentList args;

	args["factor"] = QVariant((float)65535);
	REGISTER_FILTER_WITH_ARGS("integer_default", Picviz::PVPlottingFilterAllDivide, args);

	args["factor"] = QVariant((float)PICVIZ_IPV4_MAXVAL);
	REGISTER_FILTER_WITH_ARGS("ipv4_default", Picviz::PVPlottingFilterAllDivide, args);

	args["factor"] = QVariant((float)PICVIZ_MAXFLOAT);
	REGISTER_FILTER_WITH_ARGS("float_default", Picviz::PVPlottingFilterAllDivide, args);
}
