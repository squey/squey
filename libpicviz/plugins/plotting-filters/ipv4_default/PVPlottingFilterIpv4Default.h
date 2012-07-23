/**
 * \file PVPlottingFilterIpv4Default.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H
#define PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterIpv4Default: public PVPlottingFilter
{
public:
	float* operator()(float* value);
	QString get_human_name() const { return QString("Default"); }

	CLASS_FILTER(PVPlottingFilterIpv4Default)
};

}

#endif
