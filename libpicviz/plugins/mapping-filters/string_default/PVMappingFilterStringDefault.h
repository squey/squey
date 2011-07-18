//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H

#include <pvcore/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/atomic.h>

#include <pvcore/stdint.h>

namespace Picviz {

class PVMappingFilterStringDefault: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::nraw_table_line const& values);

	CLASS_FILTER(PVMappingFilterStringDefault)
};

}

#endif
