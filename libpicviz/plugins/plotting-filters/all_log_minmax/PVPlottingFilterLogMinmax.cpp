#include "PVPlottingFilterLogMinmax.h"
#include <math.h>

#include <omp.h>

#ifdef WIN32
#define log2f logf
#endif

Picviz::PVPlottingFilterLogMinmax::PVPlottingFilterLogMinmax(PVCore::PVArgumentList const& args):
	PVPlottingFilter()
{
	INIT_FILTER(PVPlottingFilterLogMinmax, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVPlottingFilterLogMinmax)
{
	
	PVCore::PVArgumentList args;
	/*args[PVCore::PVArgumentKey("range-min", "Minimal value (0=auto)")] = QString("0.0");
	args[PVCore::PVArgumentKey("range-max", "Maximal value (0=auto)")] = QString("0.0");*/
	return args;
}

float* Picviz::PVPlottingFilterLogMinmax::operator()(float* values)
{
	assert(values);
	assert(_dest);
	assert(_mandatory_params);

	float ymin, ymax;
	ymin = _args["range-min"].toFloat();
	ymax = _args["range-max"].toFloat();

	int64_t size = _dest_size;

	if (ymin == 0.0f) {
		Picviz::mandatory_param_map::const_iterator it_min = _mandatory_params->find(Picviz::mandatory_ymin);
		if (it_min == _mandatory_params->end()) {
			PVLOG_WARN("ymin doesn't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
			memcpy(_dest, values, size*sizeof(float));
			return _dest;
		}
		ymin = (*it_min).second.second;
	}
	if (ymax == 0.0f) {
		Picviz::mandatory_param_map::const_iterator it_max = _mandatory_params->find(Picviz::mandatory_ymax);
		if (it_max == _mandatory_params->end()) {
			PVLOG_WARN("ymax doesn't exist for an axis. Maybe the mandatory minmax mapping hasn't be run ?\n");
			memcpy(_dest, values, size*sizeof(float));
			return _dest;
		}
		ymax = (*it_max).second.second;
	}

	if (ymin > ymax) {
		std::swap(ymin, ymax);
	}
	else
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
	_expand_max = log2f(max);
	_expand_diff = _expand_max - _expand_min;
}

float Picviz::PVPlottingFilterLogMinmax::expand_plotted(float value) const
{
	return (log2f(value+_offset)-_expand_min)/_expand_diff;
}

IMPL_FILTER(Picviz::PVPlottingFilterLogMinmax)
