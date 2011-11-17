#ifndef PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVPLOTTINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterTimeDefault: public PVPlottingFilter
{
public:
	float* operator()(float* value);
	QString get_human_name() const { return QString("Default (depends on mapping)"); }

	CLASS_FILTER(PVPlottingFilterTimeDefault)
};

}

#endif
