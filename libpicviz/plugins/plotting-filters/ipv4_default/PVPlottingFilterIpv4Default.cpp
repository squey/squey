#include "PVPlottingFilterIpv4Default.h"
#include <picviz/limits.h>

#include <omp.h>

float* Picviz::PVPlottingFilterIpv4Default::operator()(float* values)
{
	assert(values);
	assert(_dest);

	int64_t size = _dest_size;
#pragma omp parallel for
	for (int64_t i = 0; i < size; i++) {
		_dest[i] = values[i] / PICVIZ_IPV4_MAXVAL;
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterIpv4Default)
