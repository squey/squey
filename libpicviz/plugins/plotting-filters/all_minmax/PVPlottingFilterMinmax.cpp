/**
 * \file PVPlottingFilterMinmax.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterMinmax.h"
#include <omp.h>

uint32_t* Picviz::PVPlottingFilterMinmax::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	const ssize_t size = _dest_size;

	Picviz::mandatory_param_map::const_iterator it_min = _mandatory_params->find(Picviz::mandatory_ymin);
	Picviz::mandatory_param_map::const_iterator it_max = _mandatory_params->find(Picviz::mandatory_ymax);
	if (it_min == _mandatory_params->end() || it_max == _mandatory_params->end()) {
		PVLOG_WARN("ymin and/or ymax don't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
		copy_mapped_to_plotted(values);
		return _dest;
	}

	if (_decimal_type == PVCore::IntegerType ||
	    _decimal_type == PVCore::UnsignedIntegerType) {

		int64_t ymin, ymax;
		ymin = (int64_t) (*it_min).second.second.storage_as_uint();
		ymax = (int64_t) (*it_max).second.second.storage_as_uint();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX>>1);
			}
			return _dest;
		}

		const int64_t diff = ymax - ymin;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			const int64_t v_tmp = ((((int64_t) values[i].storage_as_uint() - ymin)*(int64_t)(UINT_MAX))/diff);
			_dest[i] = ~((uint32_t) (v_tmp & 0x00000000FFFFFFFFULL));
		}
	}
	else {
		float ymin, ymax;
		ymin = (*it_min).second.second.storage_as_float();
		ymax = (*it_max).second.second.storage_as_float();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX>>1);
			}
			return _dest;
		}

		const float diff = ymax - ymin;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t) ((double) ((values[i].storage_as_float() - ymin)/diff) * (double)UINT_MAX));
		}
	}

	return _dest;
}

void Picviz::PVPlottingFilterMinmax::init_expand(uint32_t min, uint32_t max)
{
	_expand_min = min;
	_expand_max = max;
	_expand_diff = max-min;
}

uint32_t Picviz::PVPlottingFilterMinmax::expand_plotted(uint32_t value) const
{
	return (value-_expand_min)/(_expand_diff);
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterMinmax)
