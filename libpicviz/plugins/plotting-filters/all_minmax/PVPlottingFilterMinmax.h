/**
 * \file PVPlottingFilterMinmax.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTERMINMAX_H
#define PVFILTER_PVPLOTTINGFILTERMINMAX_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterMinmax: public PVPlottingFilter
{
public:
	float* operator()(float* value);
	void init_expand(float min, float max);
	float expand_plotted(float value) const;
	QString get_human_name() const { return QString("Min/max"); }
	bool can_expand() const { return true; }

private:
	float _expand_min;
	float _expand_max;
	float _expand_diff;

	CLASS_FILTER(PVPlottingFilterMinmax)
};

}

#endif
