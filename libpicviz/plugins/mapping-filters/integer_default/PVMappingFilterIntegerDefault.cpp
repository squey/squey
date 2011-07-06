#include "PVMappingFilterIntegerDefault.h"


float Picviz::PVMappingFilterIntegerDefault::operator()(QString const& str)
{
	return (float) str.toInt();
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIntegerDefault)
