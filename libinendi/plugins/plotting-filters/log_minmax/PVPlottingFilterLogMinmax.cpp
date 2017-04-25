/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include "PVPlottingFilterLogMinmax.h"
#include <cmath>

#include <omp.h>

template <class T>
static void compute_log_plotting(pvcop::db::array const& mapped,
                                 pvcop::db::array const& minmax,
                                 const pvcop::db::selection& invalid_selection,
                                 pvcop::core::array<Inendi::PVPlottingFilter::value_type>& dest)
{
	using value_type = Inendi::PVPlottingFilter::value_type;

	double ymin;
	double ymax;
	std::tie(ymin, ymax) = Inendi::PVPlottingFilter::extract_minmax<T>(minmax);

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
	    invalid_selection ? Inendi::PVPlottingFilter::INVALID_RESERVED_PERCENT_RANGE : 0;
	const size_t valid_offset = std::numeric_limits<value_type>::max() * invalid_range;
	const double ratio =
	    (std::numeric_limits<value_type>::max() * (1 - invalid_range)) / (std::log2(ymax / ymin));
	auto& values = mapped.to_core_array<T>();

#pragma omp parallel for
	for (size_t i = 0; i < mapped.size(); i++) {
		bool invalid = invalid_selection and invalid_selection[i];
		dest[i] = ~value_type(
		    invalid
		        ? 0
		        : ratio * (std::log2((std::max<double>(ymin, (double)values[i] + offset)) / ymin)) +
		              valid_offset);
	}
}

void Inendi::PVPlottingFilterLogMinmax::operator()(pvcop::db::array const& mapped,
                                                   pvcop::db::array const& minmax,
                                                   const pvcop::db::selection& invalid_selection,
                                                   pvcop::core::array<value_type>& dest)
{
	assert(dest);

	if (mapped.is_string()) {
		compute_log_plotting<string_index_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_int32) {
		compute_log_plotting<int32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_uint32) {
		compute_log_plotting<uint32_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_uint64) {
		compute_log_plotting<uint64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_int64) {
		compute_log_plotting<int64_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_uint128) {
		compute_log_plotting<pvcop::db::uint128_t>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_float) {
		compute_log_plotting<float>(mapped, minmax, invalid_selection, dest);
	} else if (mapped.type() == pvcop::db::type_double) {
		compute_log_plotting<double>(mapped, minmax, invalid_selection, dest);
	}
}
