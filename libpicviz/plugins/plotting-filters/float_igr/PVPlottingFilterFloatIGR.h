//! \file PVMappingFilterFloat.h
//! $Id: PVMappingFilterFloat.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVPLOTTINGFILTERFLOATIGR_H
#define PVFILTER_PVPLOTTINGFILTERFLOATIGR_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterFloatIGR: public PVPlottingFilter
{
public:
	float operator()(float v);

	CLASS_FILTER(PVPlottingFilterFloatIGR)
};

}

#endif
