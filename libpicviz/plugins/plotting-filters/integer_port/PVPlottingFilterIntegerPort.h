#ifndef PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H
#define PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterIntegerPort: public PVPlottingFilter
{
public:
	float operator()(float value);

	CLASS_FILTER(PVPlottingFilterIntegerPort)
};

}

#endif
