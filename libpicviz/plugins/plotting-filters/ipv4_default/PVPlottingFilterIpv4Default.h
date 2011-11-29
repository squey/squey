#ifndef PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H
#define PVFILTER_PVPLOTTINGFILTERALLDIVIVDE_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterIpv4Default: public PVPlottingFilter
{
public:
	float* operator()(float* value);

	CLASS_FILTER(PVPlottingFilterIpv4Default)
};

}

#endif
