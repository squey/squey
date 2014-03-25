/**
 * \file PVMappingFilterDateDay.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVMappingFilterDateDay.h"

#include <unicode/ucal.h>

Picviz::PVMappingFilterDateDay::PVMappingFilterDateDay()
{
	QStringList sl("E");
	set_time_format(sl);
	set_time_symbol(UCAL_DAY_OF_WEEK);
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterDateDay)
