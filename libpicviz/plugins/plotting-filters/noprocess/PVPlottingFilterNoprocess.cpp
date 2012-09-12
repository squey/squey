/**
 * \file PVPlottingFilterNoprocess.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPlottingFilterNoprocess.h"


uint32_t* Picviz::PVPlottingFilterNoprocess::operator()(mapped_decimal_storage_type const* values)
{
	assert(values);
	assert(_dest);

	copy_mapped_to_plotted(values);
	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVPlottingFilterNoprocess)
