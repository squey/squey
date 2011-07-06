#include "PVMappingFilterFloatDefault.h"


float Picviz::PVMappingFilterFloatDefault::operator()(QString const& str)
{
	return str.toFloat();
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatDefault)
