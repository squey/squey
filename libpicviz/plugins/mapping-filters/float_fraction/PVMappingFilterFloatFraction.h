//! \file PVMappingFilterFloat.h
//! $Id: PVMappingFilterFloat.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterFloatFraction: public PVMappingFilter
{
public:
	float operator()(QString const& str);

	CLASS_FILTER(PVMappingFilterFloatFraction)
};

}

#endif
