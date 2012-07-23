/**
 * \file PVPlottingFilterNoprocess.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterNoprocess.h"


float Picviz::PVPlottingFilterNoprocess::operator()(float value)
{
	return value;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterNoprocess)
