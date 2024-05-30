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

#include <pvkernel/core/PVLogger.h>
#include "PVScalingFilterMinmax.h"
#include <omp.h>

using scaling_t = Squey::PVScalingFilter::value_type;

template <class T>
static void compute_minmax_scaling(pvcop::db::array const& mapped,
                                    pvcop::db::array const& minmax,
                                    const pvcop::db::selection& invalid_selection,
                                    pvcop::core::array<Squey::PVScalingFilter::value_type>& dest)
{
	double ymin;
	double ymax;
	std::tie(ymin, ymax) = Squey::PVScalingFilter::extract_minmax<T>(minmax);

	if (ymin == ymax) {
		const scaling_t mid = std::numeric_limits<scaling_t>::max() / 2;
		for (size_t i = 0; i < mapped.size(); i++) {
			dest[i] = invalid_selection and invalid_selection[i] ? ~scaling_t(0) : mid;
		}
		return;
	}
	assert(ymax > ymin);

	// Use double to compute value to avoid rounding issue.
	// eg: if we use only uint32_t, with ymax - ymin > uint32_max / 2, ratio will be 1 and
	// no scaling will be applied
	const double invalid_range =
	    invalid_selection ? Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const size_t valid_offset = std::numeric_limits<scaling_t>::max() * invalid_range;
	const double ratio =
	    (std::numeric_limits<scaling_t>::max() * (1 - invalid_range)) / (ymax - ymin);
	auto& values = mapped.to_core_array<T>();

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		bool invalid = invalid_selection and invalid_selection[i];
		dest[i] = ~scaling_t(invalid ? 0 : (Squey::extract_value(values[i]) - ymin) * ratio +
		                                        valid_offset);
	}
}

void Squey::PVScalingFilterMinmax::operator()(pvcop::db::array const& mapped,
                                                pvcop::db::array const& minmax,
                                                const pvcop::db::selection& invalid_selection,
                                                pvcop::core::array<value_type>& dest)
{
	assert(dest);

	if (mapped.is_string()) {
		compute_minmax_scaling<string_index_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int8") {
		compute_minmax_scaling<int8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint8") {
		compute_minmax_scaling<uint8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int16") {
		compute_minmax_scaling<int16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint16") {
		compute_minmax_scaling<uint16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int32") {
		compute_minmax_scaling<int32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint32" or
	           mapped.type() == "ipv4") {
		compute_minmax_scaling<uint32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint64" or mapped.type() == "datetime" or mapped.type() == "datetime_ms") {
		compute_minmax_scaling<uint64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "datetime_us") {
		compute_minmax_scaling<boost::posix_time::ptime>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int64") {
		compute_minmax_scaling<int64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint128" or mapped.type() == "ipv6") {
		compute_minmax_scaling<pvcop::db::uint128_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_float") {
		compute_minmax_scaling<float>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_double") {
		compute_minmax_scaling<double>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "duration") {
		compute_minmax_scaling<boost::posix_time::time_duration>(mapped, minmax, invalid_selection,
		                                                          dest);
	}
}
