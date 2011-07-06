#ifndef PVFILTER_PVPLOTTINGFILTERNOPROCESS_H
#define PVFILTER_PVPLOTTINGFILTERNOPROCESS_H

#include <pvcore/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class LibExport PVPlottingFilterNoprocess: public PVPlottingFilter
{
public:
	float operator()(float value);

	CLASS_FILTER(PVPlottingFilterNoprocess)
};

}

#endif
