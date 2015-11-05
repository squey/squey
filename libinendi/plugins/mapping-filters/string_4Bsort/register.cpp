/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMappingFilterString4Bsort.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	// Register under the name "type_format"
	REGISTER_CLASS("string_4Bsort", Inendi::PVMappingFilterString4Bsort);
}
