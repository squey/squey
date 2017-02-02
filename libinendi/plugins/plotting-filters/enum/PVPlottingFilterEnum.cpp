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

#include <pvcop/core/selected_array.h>

void Inendi::PVPlottingFilterEnum::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              const pvcop::db::selection& invalid_selection,
                                              pvcop::core::array<value_type>& dest)
{
	pvcop::db::groups groups;
	pvcop::db::extents extents;
	mapped.group(groups, extents);

	// Sort extents
	mapped.parallel_sort(extents);
	pvcop::db::indexes indexes = extents.parallel_sort();
	auto& sorted_extents = indexes.to_core_array();
	auto& core_extents = extents.to_core_array();
	auto& core_groups = groups.to_core_array();

	if (extents.size() == 1) {
		const value_type mid = std::numeric_limits<value_type>::max() / 2;
		for (size_t i = 0; i < mapped.size(); i++) {
			dest[i] = invalid_selection and invalid_selection[i] ? ~uint32_t(0) : mid;
		}
		return;
	}

	const size_t distinct_count =
	    extents.size() - (extents.has_invalid()
	                          ? (pvcop::core::algo::bit_count(extents.invalid_selection()) - 1)
	                          : 0);
	const double invalid_range = Inendi::PVPlottingFilter::INVALID_RESERVED_PERCENT_RANGE;
	const size_t valid_offset =
	    invalid_selection ? std::numeric_limits<value_type>::max() * invalid_range : 0;
	const double ratio =
	    (std::numeric_limits<value_type>::max() * (1 - (invalid_selection ? invalid_range : 0))) /
	    ((double)distinct_count - 1);

#pragma omp parallel for
	for (size_t row = 0; row < mapped.size(); row++) {
		bool invalid = invalid_selection && invalid_selection[row];
		dest[row] =
		    ~value_type(invalid ? 0 : (sorted_extents[core_groups[row]] * ratio + valid_offset));
	}
}
