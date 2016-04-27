/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVMandatoryMappingFilterMinmax.h"

// This method will be called by libinendi
// This mapping filter will be registered as a mandatory mapping filter ! (FilterT is set to
// PVMandatoryMappingFilter in PVMandatoryMappingFilter.h)
LibCPPExport void register_class()
{
	REGISTER_CLASS("mandatory_minmax", Inendi::PVMandatoryMappingFilterMinmax);
}
