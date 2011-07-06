#include "PVPlottingFilterNoprocess.h"


float Picviz::PVPlottingFilterNoprocess::operator()(float value)
{
	return value;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterNoprocess)
