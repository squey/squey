/**
 * \file PVPlottingFilterTimeDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterTimeDefault.h"
#include <picviz/limits.h>

#include <omp.h>

uint32_t* Picviz::PVPlottingFilterTimeDefault::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);


	uint32_t const* vint = &values->storage_as_uint();

	ssize_t size = _dest_size;
	uint64_t ymin,ymax;
	if (!_mapping_mode.compare("default")) {
		Picviz::mandatory_param_map::const_iterator it_min = _mandatory_params->find(Picviz::mandatory_ymin);
		Picviz::mandatory_param_map::const_iterator it_max = _mandatory_params->find(Picviz::mandatory_ymax);
		if (it_min == _mandatory_params->end() || it_max == _mandatory_params->end()) {
			PVLOG_WARN("ymin and/or ymax don't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
			memcpy(_dest, values, size*sizeof(float));
			return _dest;
		}
		const uint64_t ymin = (*it_min).second.second.storage_as_uint();
		const uint64_t ymax = (*it_max).second.second.storage_as_uint();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX>>1);
			}
			return _dest;
		}
	}
	else {
		ymin = 0;
		if (_mapping_mode.compare("24h") == 0) {
			ymax = PICVIZ_TIME_24H_MAX;
		}
		else
		if (_mapping_mode.compare("week") == 0) {
			ymax = PICVIZ_TIME_WEEK_MAX;
		}
		else
		if (_mapping_mode.compare("month") == 0) {
			ymax = PICVIZ_TIME_MONTH_MAX;
		}
		else {
			PVLOG_ERROR("(mapping: time-default) unknown time mapping mode '%s' !\n", qPrintable(_mapping_mode));
			ymax = UINT_MAX;
		}
	}

	const uint64_t ydiff = ymax-ymin;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = ~((uint32_t) (((uint64_t)(vint[i] - ymin)*(uint64_t)UINT_MAX)/ydiff));
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterTimeDefault)
