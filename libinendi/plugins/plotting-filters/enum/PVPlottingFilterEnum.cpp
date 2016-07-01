/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterEnum.h"

void Inendi::PVPlottingFilterEnum::
operator()(pvcop::db::array const& mapped, pvcop::db::array const&, uint32_t* dest)
{
	assert(dest);

	pvcop::db::groups group;
	pvcop::db::extents extents;

	mapped.group(group, extents);

	auto& core_group = group.to_core_array();

	// -1 as we count "number of space between values", not "values"
	double extend_factor = std::numeric_limits<uint32_t>::max() / ((double)extents.size() - 1);
	for (size_t row = 0; row < mapped.size(); row++) {
		dest[row] = extend_factor * core_group[row];
	}
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterEnum)
