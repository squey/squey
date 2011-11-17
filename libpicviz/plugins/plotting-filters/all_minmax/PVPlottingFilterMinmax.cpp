#include "PVPlottingFilterMinmax.h"
#include <omp.h>

float* Picviz::PVPlottingFilterMinmax::operator()(float* values)
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

#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = (values[i] - ymin) / (ymax - ymin);
	}

	return _dest;
}

void Picviz::PVPlottingFilterMinmax::init_expand(float min, float max)
{
	_expand_min = min;
	_expand_max = max;
	_expand_diff = max-min;
}

float Picviz::PVPlottingFilterMinmax::expand_plotted(float value) const
{
	return (value-_expand_min)/(_expand_diff);
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterMinmax)
