//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "PVScalingFilterEnum.h"

#include <limits>
#include <fstream>

#include <pvkernel/core/squey_bench.h>

#include <pvcop/core/selected_array.h>

void Squey::PVScalingFilterEnum::operator()(pvcop::db::array const& mapped,
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
	auto& core_groups = groups.to_core_array();

	if (extents.size() == 1) {
		const value_type mid = std::numeric_limits<value_type>::max() / 2;
		for (size_t i = 0; i < mapped.size(); i++) {
			dest[i] = invalid_selection and invalid_selection[i] ? ~uint32_t(0) : mid;
		}
		return;
	}

	const size_t distinct_count =
	    extents.size() -
	    (extents.has_invalid() ? (pvcop::core::algo::bit_count(extents.invalid_selection())) : 0);
	const double invalid_range = Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE;
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
