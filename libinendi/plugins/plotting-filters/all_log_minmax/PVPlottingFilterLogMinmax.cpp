/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVPlottingFilterLogMinmax.h"
#include <cmath>

#include <omp.h>

Inendi::PVPlottingFilterLogMinmax::PVPlottingFilterLogMinmax(PVCore::PVArgumentList const& args):
	PVPlottingFilter()
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

uint32_t* Inendi::PVPlottingFilterLogMinmax::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	const ssize_t size = _dest_size;

	Inendi::mandatory_param_map::const_iterator it_min = _mandatory_params->find(Inendi::mandatory_ymin);
	Inendi::mandatory_param_map::const_iterator it_max = _mandatory_params->find(Inendi::mandatory_ymax);
	if (it_min == _mandatory_params->end() || it_max == _mandatory_params->end()) {
		PVLOG_WARN("ymin and/or ymax don't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
		copy_mapped_to_plotted(values);
		return _dest;
	}
	
	if (_decimal_type == PVCore::IntegerType) {
		int64_t ymin, ymax;
		ymin = (*it_min).second.second.storage_as_int();
		ymax = (*it_max).second.second.storage_as_int();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX>>1);
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
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t) (((log2((int64_t)(values[i].storage_as_int())+offset) - log_ymin) / div)*((double)(UINT_MAX))));
		}
	}
	else
	if (_decimal_type == PVCore::UnsignedIntegerType) {
		int64_t ymin, ymax;
		ymin = (*it_min).second.second.storage_as_uint();
		ymax = (*it_max).second.second.storage_as_uint();

		if (ymin == ymax) {
			for (int64_t i = 0; i < size; i++) {
				_dest[i] = ~(UINT_MAX>>1);
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
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t) (((log2((int64_t)(values[i].storage_as_uint())+offset) - log_ymin) / div)*((double)(UINT_MAX))));
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

		float offset = 0;
		if (ymin <= 0) {
			offset = -ymin + 1.0f;
			ymin += offset;
			ymax += offset;
		}

		const double log_ymin = log2(ymin);
		const double div = log2(ymax) - log_ymin;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			_dest[i] = ~((uint32_t) (((log2(values[i].storage_as_float()+offset) - log_ymin) / div)*((double)(UINT_MAX))));
		}
	}


	return _dest;
}

void Inendi::PVPlottingFilterLogMinmax::init_expand(uint32_t min, uint32_t max)
{
	if (min <= 0) {
		_offset = -min + 1;
		min += _offset;
		max += _offset;
	}
	else {
		_offset = 0;
	}
	_expand_min = log2(min);
	_expand_max = log2(max);
	_expand_diff = _expand_max - _expand_min;
}

uint32_t Inendi::PVPlottingFilterLogMinmax::expand_plotted(uint32_t value) const
{
	return ~((uint32_t) (((log2(value+_offset)-_expand_min)/_expand_diff)*(double)(UINT_MAX)));
}

IMPL_FILTER(Inendi::PVPlottingFilterLogMinmax)