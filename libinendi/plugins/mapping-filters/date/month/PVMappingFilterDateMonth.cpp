/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterDateMonth.h"

#include <unicode/ucal.h>

Inendi::PVMappingFilterDateMonth::PVMappingFilterDateMonth()
{
	QStringList sl("M");
	set_time_format(sl);
	set_time_symbol(UCAL_MONTH);
}

IMPL_FILTER_NOPARAM(Inendi::PVMappingFilterDateMonth)
