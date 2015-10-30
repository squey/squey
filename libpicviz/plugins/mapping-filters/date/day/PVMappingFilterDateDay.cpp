/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
