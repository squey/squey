// Register the plugin in PVFilterLibrary
//

#include <pvfilter/PVFilterLibrary.h>
#include "PVMandatoryMappingFilterMinmax.h"

// This method will be called by libpicviz
// This mapping filter will be registered as a mandatory mapping filter ! (FilterT is set to PVMandatoryMappingFilter in PVMandatoryMappingFilter.h)
LibCPPExport void register_filter()
{
	REGISTER_FILTER("mandatory_minmax", Picviz::PVMandatoryMappingFilterMinmax);
}
