/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterEnum.h"

#include <limits>

#include <fstream>

#include <pvkernel/core/inendi_bench.h>

using plotting_t = uint32_t;

void Inendi::PVPlottingFilterEnum::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              pvcop::core::array<uint32_t>& dest)
{
	pvcop::db::groups groups;
	pvcop::db::extents extents;
	mapped.group(groups, extents);

	// Sort extents
	mapped.parallel_sort(extents);
	pvcop::db::indexes indexes = pvcop::db::indexes::parallel_sort(extents);
	auto& sorted_extents = indexes.to_core_array();

	auto& core_groups = groups.to_core_array();

	if (extents.size() == 1) {
		const plotting_t default_value = std::numeric_limits<plotting_t>::max() / 2;
		std::fill_n(dest.begin(), mapped.size(), default_value);
	} else {
		// -1 as we count "number of space between values", not "values"
		const double extend_factor =
		    std::numeric_limits<plotting_t>::max() / ((double)extents.size() - 1);
#pragma omp parallel for
		for (size_t row = 0; row < mapped.size(); row++) {
			dest[row] = ~uint32_t(extend_factor * sorted_extents[core_groups[row]]);
		}
	}
}
