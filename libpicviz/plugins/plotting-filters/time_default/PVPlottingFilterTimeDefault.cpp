/**
 * \file PVPlottingFilterTimeDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterTimeDefault.h"
#include <picviz/limits.h>

#include <omp.h>

float* Picviz::PVPlottingFilterTimeDefault::operator()(float* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	int64_t size = _dest_size;
	if (!_mapping_mode.compare("default")) {
		float ymin, ymax;

		Picviz::mandatory_param_map::const_iterator it_min = _mandatory_params->find(Picviz::mandatory_ymin);
		Picviz::mandatory_param_map::const_iterator it_max = _mandatory_params->find(Picviz::mandatory_ymax);
		if (it_min == _mandatory_params->end() || it_max == _mandatory_params->end()) {
			PVLOG_WARN("ymin and/or ymax don't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
			memcpy(_dest, values, size*sizeof(float));
			return _dest;
		}
		ymin = (*it_min).second.second;
		ymax = (*it_max).second.second;

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = 0.5;
			}
			return _dest;
		}

#pragma omp parallel for
		for (int64_t i = 0; i < size; i++) {
			_dest[i] = (values[i] - ymin) / (ymax - ymin);
		}

		return _dest;
	}

	float factor = 1.0;
	if (_mapping_mode.compare("24h") == 0) {
		factor = PICVIZ_TIME_24H_MAX;
	}
	else
	if (_mapping_mode.compare("week") == 0) {
		factor = PICVIZ_TIME_WEEK_MAX;
	}
	else
	if (_mapping_mode.compare("month") == 0) {
		factor = PICVIZ_TIME_MONTH_MAX;
	}
	else {
		PVLOG_ERROR("(mapping: time-default) unknown time mapping mode '%s' !\n", qPrintable(_mapping_mode));
	}

#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = values[i] / factor;
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterTimeDefault)
