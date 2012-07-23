/**
 * \file PVMappingFilterFloatFraction.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterFloatFraction: public PVMappingFilter
{
public:
	float operator()(QString const& str);
	QString get_human_name() const { return QString("Fraction (x/y) or classical"); }

	CLASS_FILTER(PVMappingFilterFloatFraction)
};

}

#endif
