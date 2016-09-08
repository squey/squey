/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include "PVFieldConverterSubstitution.h"
#include "PVFieldConverterSubstitutionParamWidget.h"

#include <pvkernel/core/PVClassLibrary.h>

#include <pvbase/export.h>

// This method will be called by libinendi
LibCPPExport void register_class()
{
	REGISTER_CLASS("substitution", PVFilter::PVFieldConverterSubstitution);
	REGISTER_CLASS_AS("converter_substitution", PVFilter::PVFieldConverterSubstitution,
	                  PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("substitution", PVFilter::PVFieldConverterSubstitutionParamWidget);
}
