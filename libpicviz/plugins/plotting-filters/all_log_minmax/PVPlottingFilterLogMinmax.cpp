#include "PVPlottingFilterLogMinmax.h"
#include <math.h>

#include <omp.h>

float* Picviz::PVPlottingFilterLogMinmax::operator()(float* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	float ymin, ymax;
	int64_t size = _dest_size;

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

	float offset = 0;
	if (ymin <= 0) {
		offset = -ymin + 1.0f;
		ymin += offset;
		ymax += offset;
	}

	float log_ymin = log2f(ymin);
	float div = log2f(ymax) - log_ymin;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = (log2f(values[i]+offset) - log_ymin) / div;
	}

	return _dest;
}

void Picviz::PVPlottingFilterLogMinmax::init_expand(float min, float max)
{
	if (min <= 0) {
		_offset = -min + 1.0f;
		min += _offset;
		max += _offset;
	}
	else {
		_offset = 0;
	}
	_expand_min = log2f(min);
	_expand_max = max;
	_expand_diff = log2f(max) - log2f(min);
}

float Picviz::PVPlottingFilterLogMinmax::expand_plotted(float value) const
{
	return (log2f(value+_offset)-_expand_min)/_expand_diff;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterLogMinmax)
