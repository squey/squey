/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterNoprocess.h"

uint32_t* Inendi::PVPlottingFilterNoprocess::operator()(pvcop::db::array const& mapped)
{
	assert(_dest);

	copy_mapped_to_plotted(mapped);
	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterNoprocess)
