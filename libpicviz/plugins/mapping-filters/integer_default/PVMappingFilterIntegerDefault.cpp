/**
 * \file PVMappingFilterIntegerDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterIntegerDefault.h"


float Picviz::PVMappingFilterIntegerDefault::operator()(QString const& str)
{
	return (float) str.toInt();
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIntegerDefault)
