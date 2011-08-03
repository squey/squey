#ifndef PVFILTER_PVMAPPINGFILTERUSERDEFINEDDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERUSERDEFINEDDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include <QString>

namespace Picviz {

class PVMappingFilterUserdefinedDefault: public PVMappingFilter
{
public:
	float operator()(QString const& str);

	CLASS_FILTER(PVMappingFilterUserdefinedDefault)
};

}

#endif
