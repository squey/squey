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

	CLASS_FILTER(PVPlottingFilterMinmax)
};

}

#endif
