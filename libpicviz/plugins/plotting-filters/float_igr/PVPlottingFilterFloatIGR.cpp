#include "PVPlottingFilterFloatIGR.h"
#define RANGE 15.0f

float Picviz::PVPlottingFilterFloatIGR::operator()(float f)
{
	float ret = f;
	if (ret < -RANGE) {
		PVLOG_INFO("Value too low: %0.4f\n", ret);
		ret = -RANGE;
	}
	else
	if (ret > RANGE) {
		PVLOG_INFO("Value too high: %0.4f\n", ret);
		ret = RANGE;
	}

	return (ret + RANGE)/(2*RANGE);
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterFloatIGR)
