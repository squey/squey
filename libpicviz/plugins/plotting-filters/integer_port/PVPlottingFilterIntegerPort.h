#ifndef PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H
#define PVFILTER_PVPLOTTINGFILTERINTEGERPORT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPlottingFilter.h>

namespace Picviz {

class PVPlottingFilterIntegerPort: public PVPlottingFilter
{
public:
	float operator()(float value);
	QString get_human_name() const { return QString("TCP/UDP port"); }

	CLASS_FILTER(PVPlottingFilterIntegerPort)
};

}

#endif
