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
#include <vector>

#include <pvkernel/core/squey_bench.h>

#include <pvcop/core/selected_array.h>

using value_type = Squey::PVScalingFilter::value_type;

/**
 * Spread the distinct values the selection holds over the whole axis.
 *
 * Uniform scaling gives every distinct value a slot of its own, so restricting
 * it to a selection means counting only the values the selection holds: those
 * take the slots, and the rest are placed against them. A value between two
 * selected ones sits halfway between their slots; one below every selected value
 * goes to the bottom of the axis and one above them all to the top, which is how
 * the other filters pin what falls outside their bounds.
 *
 * @return false when the selection holds no orderable value, leaving the caller
 *         to scale over the whole column
 */
static bool spread_selected_values(const pvcop::core::array<pvcop::db::index_t>& core_groups,
                                   const pvcop::core::array<pvcop::db::index_t>& sorted_extents,
                                   size_t group_count,
                                   const pvcop::db::selection& invalid_selection,
                                   const pvcop::db::selection& domain_selection,
                                   pvcop::core::array<value_type>& dest)
{
	const size_t row_count = core_groups.size();

	std::vector<bool> selected(group_count, false);
	size_t selected_count = 0;
	for (size_t row = 0; row < row_count; row++) {
		if (not domain_selection[row] or (invalid_selection and invalid_selection[row])) {
			continue;
		}
		const auto group = core_groups[row];
		if (not selected[group]) {
			selected[group] = true;
			selected_count++;
		}
	}

	if (selected_count == 0) {
		return false;
	}

	// The group holding each rank, sorted_extents being the other way round: it
	// gives the rank of a group. Walking ranks in order is what tells a value
	// whether the selected ones are below it, above it, or on both sides.
	std::vector<pvcop::db::index_t> group_at_rank(group_count);
	for (size_t group = 0; group < group_count; group++) {
		group_at_rank[sorted_extents[group]] = group;
	}

	const double invalid_range =
	    invalid_selection ? Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const auto max_value = std::numeric_limits<value_type>::max();
	const size_t valid_offset = max_value * invalid_range;

	std::vector<value_type> position(group_count);
	size_t seen = 0;

	if (selected_count == 1) {
		// A single value has no spread of its own; it takes the middle of the axis,
		// as a column of one value does, and the rest fall on either side of it.
		const auto middle = (value_type)(valid_offset + (max_value - valid_offset) / 2);
		for (size_t rank = 0; rank < group_count; rank++) {
			const auto group = group_at_rank[rank];
			if (selected[group]) {
				position[group] = middle;
				seen = 1;
			} else {
				position[group] = seen == 0 ? (value_type)valid_offset : max_value;
			}
		}
	} else {
		const double ratio = (max_value * (1 - invalid_range)) / ((double)selected_count - 1);
		for (size_t rank = 0; rank < group_count; rank++) {
			const auto group = group_at_rank[rank];
			double slot;
			if (selected[group]) {
				slot = (double)seen;
				seen++;
			} else {
				// Halfway below the selected value that comes next, which for a value
				// under every selected one is below the axis and for one above them all
				// is past its top: pinned to the ends either way.
				slot = (double)seen - 0.5;
			}
			position[group] =
			    Squey::PVScalingFilter::clamp_to_axis(slot * ratio + valid_offset, valid_offset);
		}
	}

#pragma omp parallel for
	for (size_t row = 0; row < row_count; row++) {
		const bool invalid = invalid_selection and invalid_selection[row];
		dest[row] = ~(invalid ? value_type(0) : position[core_groups[row]]);
	}

	return true;
}

void Squey::PVScalingFilterEnum::operator()(pvcop::db::array const& mapped,
                                              pvcop::db::array const&,
                                              const pvcop::db::selection& invalid_selection,
                                              const pvcop::db::selection& domain_selection,
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

	if (domain_selection and spread_selected_values(core_groups, sorted_extents, extents.size(),
	                                                invalid_selection, domain_selection, dest)) {
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
