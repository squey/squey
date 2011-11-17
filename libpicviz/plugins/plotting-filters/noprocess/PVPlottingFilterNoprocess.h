#ifndef PVFILTER_PVPLOTTINGFILTERNOPROCESS_H
#define PVFILTER_PVPLOTTINGFILTERNOPROCESS_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterNoprocess: public PVPlottingFilter
{
public:
	float operator()(float value);
	QString get_human_name() const { return QString("Default"); }

	CLASS_FILTER(PVPlottingFilterNoprocess)
};

}

#endif
