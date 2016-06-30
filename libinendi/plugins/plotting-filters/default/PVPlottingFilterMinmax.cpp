/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include "PVPlottingFilterMinmax.h"
#include <omp.h>

template <class T>
static void compute_minmax_plotting(pvcop::db::array const& mapped,
                                    pvcop::db::array const& minmax,
                                    uint32_t* dest)
{
	auto& mm = minmax.to_core_array<T>();
	T ymin = mm[0];
	T ymax = mm[1];

	if (ymin == ymax) {
		for (size_t i = 0; i < mm.size(); i++) {
			dest[i] = 0x80000000;
		}
		return;
	}
	assert(ymax > ymin);

	// Use int64 type to avoid overflow with ymax = int32_max and ymin = int32_min
	const uint32_t ratio =
	    std::numeric_limits<uint32_t>::max() / uint32_t((int64_t)ymax - (int64_t)ymin);
	auto& values = mapped.to_core_array<T>();
#pragma omp parallel for
	for (size_t i = 0; i < values.size(); i++) {
		dest[i] = uint32_t((int64_t)values[i] - (int64_t)ymin) * ratio;
	}
}

uint32_t* Inendi::PVPlottingFilterMinmax::operator()(pvcop::db::array const& mapped,
                                                     pvcop::db::array const& minmax)
{
	assert(_dest);

	if (mapped.type() == pvcop::db::type_uint32) {
		compute_minmax_plotting<uint32_t>(mapped, minmax, _dest);
	} else if (mapped.type() == pvcop::db::type_int32) {
		compute_minmax_plotting<int32_t>(mapped, minmax, _dest);
	} else {
		compute_minmax_plotting<float>(mapped, minmax, _dest);
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterMinmax)
