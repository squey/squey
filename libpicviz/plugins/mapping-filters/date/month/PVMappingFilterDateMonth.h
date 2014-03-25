/**
 * \file PVMappingFilterDateMonth.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVMAPPINGFILTERDATEMONTH_H
#define PVFILTER_PVMAPPINGFILTERDATEMONTH_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include "../PVMappingFilterDateBase.h"

namespace Picviz {

class PVMappingFilterDateMonth : public PVMappingFilterDateBase
{
public:
	PVMappingFilterDateMonth();

protected:
	QString get_human_name() const override { return QString("Month"); }

	CLASS_REGISTRABLE(PVMappingFilterDateMonth)
};

}

#endif // PVFILTER_PVMAPPINGFILTERDATEMONTH_H
