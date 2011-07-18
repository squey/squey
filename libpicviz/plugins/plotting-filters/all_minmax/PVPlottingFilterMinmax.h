#ifndef PVFILTER_PVPLOTTINGFILTERMINMAX_H
#define PVFILTER_PVPLOTTINGFILTERMINMAX_H

#include <pvcore/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterMinmax: public PVPlottingFilter
{
public:
	float* operator()(float* value);

	CLASS_FILTER(PVPlottingFilterMinmax)
};

}

#endif
