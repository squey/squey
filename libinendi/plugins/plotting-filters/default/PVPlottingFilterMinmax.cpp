/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include "PVPlottingFilterMinmax.h"
#include <omp.h>

#include <boost/multiprecision/cpp_int.hpp>

using plotting_t = Inendi::PVPlottingFilter::value_type;

template <class T>
static void compute_minmax_plotting(pvcop::db::array const& mapped,
                                    pvcop::db::array const& minmax,
                                    const pvcop::db::selection& invalid_selection,
                                    pvcop::core::array<Inendi::PVPlottingFilter::value_type>& dest)
{
	double ymin;
	double ymax;
	std::tie(ymin, ymax) = Inendi::PVPlottingFilter::extract_minmax<T>(minmax);

	if (ymin == ymax) {
		const plotting_t mid = std::numeric_limits<plotting_t>::max() / 2;
		for (size_t i = 0; i < mapped.size(); i++) {
			dest[i] = invalid_selection and invalid_selection[i] ? ~plotting_t(0) : mid;
		}
		return;
	}
	assert(ymax > ymin);

	// Use double to compute value to avoid rounding issue.
	// eg: if we use only uint32_t, with ymax - ymin > uint32_max / 2, ratio will be 1 and
	// no scaling will be applied
	const double invalid_range =
	    invalid_selection ? Inendi::PVPlottingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const size_t valid_offset = std::numeric_limits<plotting_t>::max() * invalid_range;
	const double ratio =
	    (std::numeric_limits<plotting_t>::max() * (1 - invalid_range)) / (ymax - ymin);
	auto& values = mapped.to_core_array<T>();

#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		bool invalid = invalid_selection and invalid_selection[i];
		dest[i] = ~plotting_t(invalid ? 0 : (Inendi::extract_value(values[i]) - ymin) * ratio +
		                                        valid_offset);
	}
}

void Inendi::PVPlottingFilterMinmax::operator()(pvcop::db::array const& mapped,
                                                pvcop::db::array const& minmax,
                                                const pvcop::db::selection& invalid_selection,
                                                pvcop::core::array<value_type>& dest)
{
	assert(dest);

	if (mapped.is_string()) {
		compute_minmax_plotting<string_index_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int8") {
		compute_minmax_plotting<int8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint8") {
		compute_minmax_plotting<uint8_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int16") {
		compute_minmax_plotting<int16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint16") {
		compute_minmax_plotting<uint16_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int32") {
		compute_minmax_plotting<int32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint32" or mapped.type() == "datetime" or
	           mapped.type() == "ipv4") {
		compute_minmax_plotting<uint32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint64" or mapped.type() == "datetime_ms" or
	           mapped.type() == "datetime_us") {
		compute_minmax_plotting<uint64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_int64") {
		compute_minmax_plotting<int64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_uint128" or mapped.type() == "ipv6") {
		compute_minmax_plotting<pvcop::db::uint128_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_float") {
		compute_minmax_plotting<float>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "number_double") {
		compute_minmax_plotting<double>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == "duration") {
		compute_minmax_plotting<boost::posix_time::time_duration>(mapped, minmax, invalid_selection,
		                                                          dest);
	}
}
