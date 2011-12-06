//! \file PVMappingFilterStringSort.h
//! $Id: PVMappingFilterStringSort.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

class PVMappingFilterHostDefault: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	QString get_human_name() const { return QString("Default"); }
	
	CLASS_FILTER(PVMappingFilterHostDefault)
};

}

#endif
