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
#include "PVScalingFilterLogMinmax.h"
#include <cmath>

#include <omp.h>

template <class T>
static void compute_log_scaling(pvcop::db::array const& mapped,
                                 pvcop::db::array const& minmax,
                                 const pvcop::db::selection& invalid_selection,
                                 pvcop::core::array<Squey::PVScalingFilter::value_type>& dest)
{
	using value_type = Squey::PVScalingFilter::value_type;

	double ymin;
	double ymax;
	std::tie(ymin, ymax) = Squey::PVScalingFilter::extract_minmax<T>(minmax);

	if (ymin == ymax) {
		for (size_t i = 0; i < mapped.size(); i++) {
			dest[i] = invalid_selection and invalid_selection[i] ? ~value_type(0) : 1UL << 31;
		}
		return;
	}

	int64_t offset = 0;
	if (ymin <= 0) {
		// Ensure values are between 0 (exclude) and infinity (and beyond)
		offset = -ymin + 1;
		ymin += offset;
		ymax += offset;
	}

	const double invalid_range =
	    invalid_selection ? Squey::PVScalingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const size_t valid_offset = std::numeric_limits<value_type>::max() * invalid_range;
	const double ratio =
	    (std::numeric_limits<value_type>::max() * (1 - invalid_range)) / (std::log2(ymax / ymin));
	auto& values = mapped.to_core_array<T>();

#pragma omp parallel for
	for (size_t i = 0; i < mapped.size(); i++) {
		bool invalid = invalid_selection and invalid_selection[i];
		dest[i] = ~value_type(
		    invalid ? 0 : ratio * (std::log2((std::max<double>(
		                                         ymin, Squey::extract_value(values[i]) + offset)) /
		                                     ymin)) +
		                      valid_offset);
	}
}

void Squey::PVScalingFilterLogMinmax::operator()(pvcop::db::array const& mapped,
                                                   pvcop::db::array const& minmax,
                                                   const pvcop::db::selection& invalid_selection,
                                                   pvcop::core::array<value_type>& dest)
{
	assert(dest);

	if (mapped.is_string()) {
		compute_log_scaling<string_index_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int8") {
		compute_log_scaling<int8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint8") {
		compute_log_scaling<uint8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int16") {
		compute_log_scaling<int16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint16") {
		compute_log_scaling<uint16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int32") {
		compute_log_scaling<int32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint32" or
	           mapped.type() == "ipv4") {
		compute_log_scaling<uint32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint64" or mapped.type() == "datetime_us" or
	           mapped.type() == "datetime" or mapped.type() == "datetime_ms") {
		compute_log_scaling<uint64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int64") {
		compute_log_scaling<int64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint128" or mapped.type() == "ipv6") {
		compute_log_scaling<pvcop::db::uint128_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_float") {
		compute_log_scaling<float>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_double") {
		compute_log_scaling<double>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "duration") {
		compute_log_scaling<boost::posix_time::time_duration>(mapped, minmax, invalid_selection,
		                                                       dest);
	}
}
