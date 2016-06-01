/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVLogger.h>
#include "PVPlottingFilterTimeDefault.h"

#include <omp.h>

static constexpr int64_t INENDI_TIME_24H_MAX = 86399;
static constexpr int64_t INENDI_TIME_WEEK_MAX =
    604793; /* 'Saturday 23:59:59'. Sun = 0, Sat = 6. (6 * 86399 = 518394) */
static constexpr int64_t INENDI_TIME_MONTH_MAX = 2678369;

uint32_t* Inendi::PVPlottingFilterTimeDefault::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	uint32_t const* vint = &values->storage_as_uint();

	ssize_t size = _dest_size;
	int64_t ymin, ymax;
	ymin = INT_MIN;
	ymax = INT_MAX;
	if (_mapping_mode.compare("default") == 0) {
		Inendi::mandatory_param_map::const_iterator it_min =
		    _mandatory_params->find(Inendi::mandatory_ymin);
		Inendi::mandatory_param_map::const_iterator it_max =
		    _mandatory_params->find(Inendi::mandatory_ymax);
		if (it_min == _mandatory_params->end() || it_max == _mandatory_params->end()) {
			PVLOG_WARN("ymin and/or ymax don't exist for an axis. Maybe the mandatory minmax "
			           "mapping hasn't be run ?\n");
			memcpy(_dest, values, size * sizeof(uint32_t));
			return _dest;
		}

		ymin = (*it_min).second.second.storage_as_int();
		ymax = (*it_max).second.second.storage_as_int();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX >> 1);
			}
			return _dest;
		}
	} else if (_mapping_mode.compare("24h") == 0) {
		ymin = 0;
		ymax = INENDI_TIME_24H_MAX;
	} else if (_mapping_mode.compare("week") == 0) {
		ymin = 0;
		ymax = INENDI_TIME_WEEK_MAX;
	} else if (_mapping_mode.compare("month") == 0) {
		ymin = 0;
		ymax = INENDI_TIME_MONTH_MAX;
	} else {
		PVLOG_ERROR("(mapping: time-default) unknown time mapping mode '%s' !\n",
		            qPrintable(_mapping_mode));
	}

	const int64_t ydiff = llabs(ymax - ymin);
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] =
		    ~((uint32_t)(((uint64_t)((int64_t)(vint[i]) - ymin) * (uint64_t)UINT_MAX) / ydiff));
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Inendi::PVPlottingFilterTimeDefault)
