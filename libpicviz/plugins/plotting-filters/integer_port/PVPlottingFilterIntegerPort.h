#ifndef PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H
#define PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H

#include <pvcore/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class LibExport PVPlottingFilterIntegerPort: public PVPlottingFilter
{
public:
	float operator()(float value);

	CLASS_FILTER(PVPlottingFilterIntegerPort)
};

}

#endif
