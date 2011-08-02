//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERTIMEWEEK_H
#define PVFILTER_PVMAPPINGFILTERTIMEWEEK_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include "../default/PVMappingFilterTimeDefault.h"

namespace Picviz {

class PVMappingFilterTimeWeek: public PVMappingFilterTimeDefault
{
protected:
	float cal_to_float(Calendar* cal, bool& success); 

	CLASS_FILTER(PVMappingFilterTimeWeek)
};

}

#endif
