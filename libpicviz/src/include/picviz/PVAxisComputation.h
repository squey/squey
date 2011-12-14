#ifndef PICVIZ_PVAXISCOMPITATION_H
#define PICVIZ_PVAXISCOMPITATION_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>

namespace Picviz {

class PVSource;

class PVAxisComputation: public PVFilter::PVFilterFunctionBase<bool, Picviz::PVSource*>, PVCore::PVRegistrableClass<PVAxisComputation>
{
public:
	QString get_human_name() const = 0;
};

}

#endif
