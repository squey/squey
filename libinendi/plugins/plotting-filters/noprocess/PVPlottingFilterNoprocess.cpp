/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterNoprocess.h"

uint32_t* Inendi::PVPlottingFilterNoprocess::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);

	copy_mapped_to_plotted(values);
	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterNoprocess)
