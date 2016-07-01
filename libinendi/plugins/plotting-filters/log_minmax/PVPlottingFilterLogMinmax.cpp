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
static void
compute_log_plotting(pvcop::db::array const& mapped, pvcop::db::array const& minmax, uint32_t* dest)
{
	auto& mm = minmax.to_core_array<T>();
	double ymin = mm[0];
	double ymax = mm[1];

	if (ymin == ymax) {
		std::fill_n(dest, mapped.size(), 1UL << 31);
		return;
	}

	int64_t offset = 0;
	if (ymin <= 0) {
		// Ensure values are between 0 (exclude) and infinity (and beyond)
		offset = -ymin + 1;
		ymin += offset;
		ymax += offset;
	}

	const double ratio = std::numeric_limits<uint32_t>::max() / (std::log2(ymax / ymin));
	auto& values = mapped.to_core_array<T>();
#pragma omp parallel for
	for (size_t i = 0; i < mapped.size(); i++) {
		dest[i] = ratio * (std::log2(((double)values[i] + offset) / ymin));
	}
}

void Inendi::PVPlottingFilterLogMinmax::
operator()(pvcop::db::array const& mapped, pvcop::db::array const& minmax, uint32_t* dest)
{
	assert(dest);

	if (mapped.type() == pvcop::db::type_int32) {
		compute_log_plotting<int32_t>(mapped, minmax, dest);
	} else if (mapped.type() == pvcop::db::type_uint32) {
		compute_log_plotting<uint32_t>(mapped, minmax, dest);
	} else {
		compute_log_plotting<float>(mapped, minmax, dest);
	}
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterLogMinmax)
