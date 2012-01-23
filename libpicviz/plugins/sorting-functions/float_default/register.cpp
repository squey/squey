// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFloatDefaultSortingFunc.h"

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("float_default", Picviz::PVFloatDefaultSortingFunc);
}
