#include "PVPlottingFilterFloatIGR.h"


float Picviz::PVPlottingFilterFloatIGR::operator()(float f)
{
	float ret = f;
	if (ret < -8.0) {
		ret = -8.0;
	}
	else
	if (ret > 8.0) {
		ret = 8.0;
	}

	return (ret + 8.0)/16.0;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterFloatIGR)
