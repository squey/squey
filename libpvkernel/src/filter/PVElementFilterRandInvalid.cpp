/**
 * \file PVElementFilterRandInvalid.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/filter/PVElementFilterRandInvalid.h>
#include <cstdlib>
#include <stdio.h>

PVFilter::PVElementFilterRandInvalid::PVElementFilterRandInvalid() :
	PVElementFilter()
{
	std::srand(time(NULL));
}

PVCore::PVElement& PVFilter::PVElementFilterRandInvalid::operator()(PVCore::PVElement &elt)
{
	bool invalidate = rand() & 1;
	if (invalidate)
		elt.set_invalid();
	return elt;
}
