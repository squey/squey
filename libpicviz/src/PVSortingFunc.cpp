/**
 * \file PVSortingFunc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVSortingFunc.h>


Picviz::PVSortingFunc::PVSortingFunc(PVCore::PVArgumentList const& l):
	PVCore::PVFunctionArgs<f_type>(l),
	PVCore::PVRegistrableClass<PVSortingFunc>()
{
}
