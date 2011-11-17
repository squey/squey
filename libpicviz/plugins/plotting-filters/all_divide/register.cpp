// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVPlottingFilterAllDivide.h"

#include <QVariant>
#include <picviz/limits.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
#if 0
	// Register under the name "type_format"
	PVCore::PVArgumentList args;

	args["factor"] = QVariant((float)65535);
	REGISTER_CLASS_WITH_ARGS("integer_default", Picviz::PVPlottingFilterAllDivide, args);

	args["factor"] = QVariant((float)PICVIZ_IPV4_MAXVAL);
	REGISTER_CLASS_WITH_ARGS("ipv4_default", Picviz::PVPlottingFilterAllDivide, args);

	args["factor"] = QVariant((float)PICVIZ_MAXFLOAT);
	REGISTER_CLASS_WITH_ARGS("float_default", Picviz::PVPlottingFilterAllDivide, args);
#endif
}
