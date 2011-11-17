//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERTIME24H_H
#define PVFILTER_PVMAPPINGFILTERTIME24H_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include "../default/PVMappingFilterTimeDefault.h"

namespace Picviz {

class PVMappingFilterTime24h: public PVMappingFilterTimeDefault
{
public:
	QString get_human_name() const { return QString("24 hours"); }
protected:
	float cal_to_float(Calendar* cal, bool& success); 

	CLASS_REGISTRABLE(PVMappingFilterTime24h)
};

}

#endif
