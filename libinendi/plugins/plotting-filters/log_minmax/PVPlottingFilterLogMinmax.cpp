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

#include <pvcop/db/algo.h>

Inendi::PVPlottingFilterLogMinmax::PVPlottingFilterLogMinmax(PVCore::PVArgumentList const& args)
    : PVPlottingFilter()
{
	INIT_FILTER(PVPlottingFilterLogMinmax, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVPlottingFilterLogMinmax)
{

	PVCore::PVArgumentList args;
	/*args[PVCore::PVArgumentKey("range-min", "Minimal value (0=auto)")] = QString("0.0");
	args[PVCore::PVArgumentKey("range-max", "Maximal value (0=auto)")] = QString("0.0");*/
	return args;
}

uint32_t* Inendi::PVPlottingFilterLogMinmax::operator()(pvcop::db::array const& mapped)
{
	assert(_dest);

	const ssize_t size = _dest_size;

	auto res = pvcop::db::algo::minmax(mapped);
	if (mapped.type() == pvcop::db::type_int32) {
		auto& mm = res.to_core_array<int32_t>();
		int64_t ymin = mm[0];
		int64_t ymax = mm[1];

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}

		int64_t offset = 0;
		if (ymin <= 0) {
			offset = -ymin + 1;
			ymin += offset;
			ymax += offset;
		}

		const double log_ymin = log2(ymin);
		const double div = log2(ymax) - log_ymin;
		auto& values = mapped.to_core_array<int32_t>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t)(((log2((int64_t)(values[i]) + offset) - log_ymin) / div) *
			                        ((double)(UINT_MAX))));
		}
	} else if (mapped.type() == pvcop::db::type_uint32) {
		// FIXME : Typing is weird
		auto& mm = res.to_core_array<uint32_t>();
		int64_t ymin = mm[0];
		int64_t ymax = mm[1];

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}

		int64_t offset = 0;
		if (ymin <= 0) {
			offset = -ymin + 1;
			ymin += offset;
			ymax += offset;
		}

		const double log_ymin = log2(ymin);
		const double div = log2(ymax) - log_ymin;
		auto& values = mapped.to_core_array<uint32_t>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t)(((log2((int64_t)(values[i]) + offset) - log_ymin) / div) *
			                        ((double)(UINT_MAX))));
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

		float offset = 0;
		if (ymin <= 0) {
			offset = -ymin + 1.0f;
			ymin += offset;
			ymax += offset;
		}

		const double log_ymin = log2(ymin);
		const double div = log2(ymax) - log_ymin;
		auto& values = mapped.to_core_array<float>();
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] =
			    ~((uint32_t)(((log2(values[i] + offset) - log_ymin) / div) * ((double)(UINT_MAX))));
		}
	}

	return _dest;
}

IMPL_FILTER(Inendi::PVPlottingFilterLogMinmax)
