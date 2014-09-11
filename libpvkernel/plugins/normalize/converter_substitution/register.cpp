/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldConverterSubstitution.h"
#include "PVFieldConverterSubstitutionParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("substitution", PVFilter::PVFieldConverterSubstitution);
	REGISTER_CLASS_AS("converter_substitution", PVFilter::PVFieldConverterSubstitution, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("substitution", PVFilter::PVFieldConverterSubstitutionParamWidget);
}
