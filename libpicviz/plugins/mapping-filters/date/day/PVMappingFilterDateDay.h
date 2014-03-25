/**
 * \file PVMappingFilterDateDay.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVMAPPINGFILTERDATEDAY_H
#define PVFILTER_PVMAPPINGFILTERDATEDAY_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include "../PVMappingFilterDateBase.h"

namespace Picviz {

class PVMappingFilterDateDay : public PVMappingFilterDateBase
{
public:
	PVMappingFilterDateDay();

protected:
	QString get_human_name() const override { return QString("Day of week"); }

	CLASS_REGISTRABLE(PVMappingFilterDateDay)
};

}

#endif // PVFILTER_PVMAPPINGFILTERDATEDAY_H
