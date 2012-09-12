/**
 * \file PVMappingFilterFloatFraction.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterFloatFraction.h"


Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterFloatFraction::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	static QChar sep1('/');
	static QChar sep2('\\');

	assert(_dest);
	assert(values.size() >= _dest_size);

	const ssize_t size = values.size();
	
#pragma omp parallel
	{
		QString str;
#pragma omp parallel for
		for (ssize_t i = 0; i < size; i++) {
			values[i].get_qstr(str);
			int index_sep = str.indexOf(sep1);
			if (index_sep == -1) {
				index_sep = str.indexOf(sep2);
				if (index_sep == -1) {
					_dest[i].storage_as_float() = str.toFloat();
					continue;
				}
			}

			const float f = str.left(index_sep).toFloat();
			const float div = str.mid(index_sep+1).toFloat();
			float res;
			if (div == 0) {
				res = f;
			}
			else {
				res = f/div;
			}
			_dest[i].storage_as_float() = res;
		}
	}

	return _dest;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatFraction)
