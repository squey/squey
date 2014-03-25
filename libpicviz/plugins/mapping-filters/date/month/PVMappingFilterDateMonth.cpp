/**
 * \file PVMappingFilterDateMonth.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVMappingFilterDateMonth.h"

#include <unicode/ucal.h>

Picviz::PVMappingFilterDateMonth::PVMappingFilterDateMonth()
{
	QStringList sl("M");
	set_time_format(sl);
	set_time_symbol(UCAL_MONTH);
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterDateMonth)
