//! \file PVMappingFilterStringSort.h
//! $Id: PVMappingFilterStringSort.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvcore/general.h>
#include <picviz/PVMappingFilter.h>

#ifdef WIN32
#include <pvcore/win32-vs2008-stdint.h>
#else
#include <stdint.h>
#endif

namespace Picviz {

class LibExport PVMappingFilterHostDefault: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::nraw_table_line const& values);
	
	CLASS_FILTER(PVMappingFilterHostDefault)
};

}

#endif
