/**
 * \file PVMappingFilterFloatDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterFloatDefault.h"


float Picviz::PVMappingFilterFloatDefault::operator()(QString const& str)
{
	return str.toFloat();
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatDefault)
