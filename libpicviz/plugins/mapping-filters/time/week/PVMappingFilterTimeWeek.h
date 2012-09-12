/**
 * \file PVMappingFilterTimeWeek.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEWEEK_H
#define PVFILTER_PVMAPPINGFILTERTIMEWEEK_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include "../default/PVMappingFilterTimeDefault.h"

namespace Picviz {

class PVMappingFilterTimeWeek: public PVMappingFilterTimeDefault
{
protected:
	int32_t cal_to_int(Calendar* cal, bool& success) override;
	QString get_human_name() const override { return QString("Week"); }

	CLASS_REGISTRABLE(PVMappingFilterTimeWeek)
};

}

#endif
