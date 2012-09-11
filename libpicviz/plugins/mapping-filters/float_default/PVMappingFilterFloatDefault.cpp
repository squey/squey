/**
 * \file PVMappingFilterFloatDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterFloatDefault.h"


Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterFloatDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	const ssize_t size = values.size();
	
#pragma omp parallel
	{
		QString stmp;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			values[i].get_qstr(stmp);
			_dest[i].storage_as_float() = stmp.toFloat();
		}
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatDefault)
