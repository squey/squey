/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include "PVMappingFilterUniform.h"

#include <limits>

#include <fstream>

#include <pvkernel/core/inendi_bench.h>

using mapping_t = uint32_t;

/*****************************************************************************
 *
 * Inendi::PVMappingFilterUniform::operator()
 *
 *****************************************************************************/

pvcop::db::array Inendi::PVMappingFilterUniform::operator()(PVCol const col,
                                                            PVRush::PVNraw const& nraw)
{
	BENCH_START(mapping);

	const auto data_array = nraw.collection().column(col);

	pvcop::db::array mapping_array(pvcop::db::type_traits::type<mapping_t>::get_type_id(),
	                               data_array.size());
	auto& mapping = mapping_array.to_core_array<mapping_t>();

	BENCH_START(group);

	pvcop::db::groups groups;
	pvcop::db::extents extents;
	data_array.group(groups, extents);

	BENCH_END(group, "PVMappingFilterUniform::group", data_array.size(),
	          pvcop::db::type_traits(data_array.type()).get_size(), mapping.size(),
	          sizeof(mapping_t));

	auto& core_extents = extents.to_core_array();

	BENCH_START(sort);

	// Sort extents
	extents.parallel_sort_with(data_array);
	pvcop::db::indexes indexes(extents.size());
	auto& sorted_extents = indexes.to_core_array();
	indexes.parallel_sort_on(extents);

	BENCH_END(sort, "PVMappingFilterUniform::sort", data_array.size(),
	          pvcop::db::type_traits(data_array.type()).get_size(), mapping.size(),
	          sizeof(mapping_t));

	auto& core_groups = groups.to_core_array();

	BENCH_START(fill_mapping);

	if (extents.size() == 1) {
		const mapping_t default_value = std::numeric_limits<mapping_t>::max() / 2;
		std::fill(mapping.begin(), mapping.end(), default_value);
	} else {
		// -1 as we count "number of space between values", not "values"
		const double extend_factor =
		    std::numeric_limits<mapping_t>::max() / ((double)extents.size() - 1);
#pragma omp parallel for
		for (size_t row = 0; row < data_array.size(); row++) {
			mapping[row] = extend_factor * sorted_extents[core_groups[row]];
		}
	}

	BENCH_END(fill_mapping, "PVMappingFilterUniform::fill_mapping", data_array.size(),
	          pvcop::db::type_traits(data_array.type()).get_size(), mapping.size(),
	          sizeof(mapping_t));

	BENCH_END(mapping, "PVMappingFilterUniform::operator()", data_array.size(),
	          pvcop::db::type_traits(data_array.type()).get_size(), mapping.size(),
	          sizeof(mapping_t));

	return mapping_array;
}
