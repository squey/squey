/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldConverterStruct.h"
#include "PVFieldConverterStructParamWidget.h"

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("struct", PVFilter::PVFieldConverterStruct);
	REGISTER_CLASS_AS("converter_struct", PVFilter::PVFieldConverterStruct,
	                  PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("struct", PVFilter::PVFieldConverterStructParamWidget);
}
