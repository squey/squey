/**
 * \file PVPlottingFilterIntegerPort.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterIntegerPort.h"


float Picviz::PVPlottingFilterIntegerPort::operator()(float value)
{
	if (value <= 1024) {
		return (value / 2048);
	} else {
		return (((value-1024) / (2*(65535-1024))) + 0.5);		
	}
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterIntegerPort)
