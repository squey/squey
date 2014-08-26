/**
 * \file register.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include "PVFieldConverterValueMapper.h"
#include "PVFieldConverterValueMapperParamWidget.h"

#include <pvkernel/rush/PVAxisTagsDec.h>

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	REGISTER_CLASS("value_mapper", PVFilter::PVFieldConverterValueMapper);
	REGISTER_CLASS_AS("converter_value_mapper", PVFilter::PVFieldConverterValueMapper, PVFilter::PVFieldsFilterReg);

	REGISTER_CLASS("value_mapper", PVFilter::PVFieldConverterValueMapperParamWidget);
}
