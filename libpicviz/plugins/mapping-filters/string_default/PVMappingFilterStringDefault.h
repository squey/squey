//! \file PVMappingFilterIPv4Default.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2012
//! Copyright (C) Philippe Saadé 2011-2012
//! Copyright (C) Picviz Labs 2012

#ifndef PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/atomic.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

class PVMappingFilterStringDefault: public PVMappingFilter
{
public:
	PVMappingFilterStringDefault(PVCore::PVArgumentList const& args = PVMappingFilterStringDefault::default_args());

public:
	float* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);

	CLASS_FILTER(PVMappingFilterStringDefault)
};

}

#endif
