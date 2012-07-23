/**
 * \file PVMappingFilterFloatFraction.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterFloatFraction.h"


float Picviz::PVMappingFilterFloatFraction::operator()(QString const& str)
{
	static QChar sep1('/');
	static QChar sep2('\\');
	int index_sep = str.indexOf(sep1);
	if (index_sep == -1) {
		index_sep = str.indexOf(sep2);
		if (index_sep == -1) {
			return str.toFloat();
		}
	}

	float f = str.left(index_sep).toFloat();
	float div = str.mid(index_sep+1).toFloat();
	if (div == 0) {
		return f;
	}
	return f/div;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterFloatFraction)
