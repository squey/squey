#ifndef PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H
#define PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterAllDivide: public PVPlottingFilter
{
public:
	PVPlottingFilterAllDivide(const PVCore::PVArgumentList& args);
	float* operator()(float* value);

	CLASS_FILTER(PVPlottingFilterAllDivide)
};

}

#endif
