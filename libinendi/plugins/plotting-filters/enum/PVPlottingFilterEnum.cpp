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

void Inendi::PVPlottingFilterEnum::
operator()(pvcop::db::array const& mapped, pvcop::db::array const&, uint32_t* dest)
{
	BENCH_START(mapping);

	BENCH_START(group);

	pvcop::db::groups groups;
	pvcop::db::extents extents;
	mapped.group(groups, extents);

	BENCH_END(group, "PVPlottingFilterEnum::group", mapped.size(),
	          pvcop::db::type_traits(mapped.type()).get_size(), mapped.size(), sizeof(plotting_t));

	auto& core_extents = extents.to_core_array();

	BENCH_START(sort);

	// Sort extents
	mapped.parallel_sort(extents);
	pvcop::db::indexes indexes = pvcop::db::indexes::parallel_sort(extents);
	auto& sorted_extents = indexes.to_core_array();
	indexes.parallel_sort(extents);

	BENCH_END(sort, "PVPlottingFilterEnum::sort", mapped.size(),
	          pvcop::db::type_traits(mapped.type()).get_size(), mapped.size(), sizeof(plotting_t));

	auto& core_groups = groups.to_core_array();

	BENCH_START(fill_mapping);

	if (extents.size() == 1) {
		const plotting_t default_value = std::numeric_limits<plotting_t>::max() / 2;
		std::fill_n(dest, mapped.size(), default_value);
	} else {
		// -1 as we count "number of space between values", not "values"
		const double extend_factor =
		    std::numeric_limits<plotting_t>::max() / ((double)extents.size() - 1);
#pragma omp parallel for
		for (size_t row = 0; row < mapped.size(); row++) {
			dest[row] = ~uint32_t(extend_factor * sorted_extents[core_groups[row]]);
		}
	}

	BENCH_END(fill_mapping, "PVPlottingFilterEnum::fill_mapping", mapped.size(),
	          pvcop::db::type_traits(mapped.type()).get_size(), mapped.size(), sizeof(plotting_t));

	BENCH_END(mapping, "PVPlottingFilterEnum::operator()", mapped.size(),
	          pvcop::db::type_traits(mapped.type()).get_size(), mapped.size(), sizeof(plotting_t));
}
