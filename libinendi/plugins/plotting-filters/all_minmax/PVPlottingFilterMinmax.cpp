/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include "PVPlottingFilterMinmax.h"
#include <omp.h>

#include <pvcop/db/algo.h>

uint32_t* Inendi::PVPlottingFilterMinmax::operator()(pvcop::db::array const& mapped)
{
	assert(_dest);

	const ssize_t size = _dest_size;

	auto res = pvcop::db::algo::minmax(mapped);
	if (mapped.type() == pvcop::db::type_uint32) {
		auto& mm = res.to_core_array<uint32_t>();
		uint32_t ymin = mm[0];
		uint32_t ymax = mm[1];

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}
		assert(ymax > ymin);

		const uint64_t diff = (uint64_t)ymax - (uint64_t)ymin;
		auto& values = mapped.to_core_array<uint32_t>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			const uint64_t v_tmp = (((int64_t)values[i] - ymin) * (int64_t)(UINT_MAX)) / diff;
			_dest[i] = ~((uint32_t)(v_tmp & 0x00000000FFFFFFFFULL));
		}
	} else if (mapped.type() == pvcop::db::type_int32) {
		auto& mm = res.to_core_array<int32_t>();
		int32_t ymin = mm[0];
		int32_t ymax = mm[1];

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}
		assert(ymax > ymin);

		const int64_t diff = (int64_t)ymax - (int64_t)ymin;
		auto& values = mapped.to_core_array<int32_t>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			const uint64_t v_tmp = (((int64_t)values[i] - ymin) * (int64_t)(UINT_MAX)) / diff;
			_dest[i] = ~((uint32_t)(v_tmp & 0x00000000FFFFFFFFULL));
		}
	} else {
		auto& mm = res.to_core_array<float>();
		float ymin = mm[0];
		float ymax = mm[1];

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}

		const float diff = ymax - ymin;
		auto& values = mapped.to_core_array<float>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t)((double)((values[i] - ymin) / diff) * (double)UINT_MAX));
		}
	}

	return _dest;
}

void Inendi::PVPlottingFilterMinmax::init_expand(uint32_t min, uint32_t max)
{
	_expand_min = min;
	_expand_max = max;
	_expand_diff = max - min;
}

uint32_t Inendi::PVPlottingFilterMinmax::expand_plotted(uint32_t value) const
{
	return (value - _expand_min) / (_expand_diff);
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterMinmax)
