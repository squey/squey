/**
 * \file PVPlottingFilterFloatIGR.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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
