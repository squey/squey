#include "PVPlottingFilterAllDivide.h"
#include <picviz/limits.h>

#include <omp.h>

Picviz::PVPlottingFilterAllDivide::PVPlottingFilterAllDivide(const PVFilter::PVArgumentList& args)
{
	INIT_FILTER(PVPlottingFilterAllDivide, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVPlottingFilterAllDivide)
{
	PVFilter::PVArgumentList args;
	args["factor"] = QVariant((float)1.0);
	return args;
}

float* Picviz::PVPlottingFilterAllDivide::operator()(float* values)
{
	assert(values);
	assert(_dest);

	float factor = _args["factor"].toFloat();

	int64_t size = _dest_size;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = values[i] / factor;
	}

	return _dest;
}

IMPL_FILTER(Picviz::PVPlottingFilterAllDivide)
