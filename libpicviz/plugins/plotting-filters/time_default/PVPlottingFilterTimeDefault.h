#ifndef PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H

#include <pvcore/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class LibExport PVPlottingFilterTimeDefault: public PVPlottingFilter
{
public:
	float* operator()(float* value);

	CLASS_FILTER(PVPlottingFilterTimeDefault)
};

}

#endif
