//! \file PVMappingFilterIPv44Bsort.h
//! $Id: PVMappingFilterIPv44Bsort.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H

#include <pvcore/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/atomic.h>

#ifdef WIN32
#include <pvcore/win32-vs2008-stdint.h>
#else
#include <stdint.h>
#endif

namespace Picviz {

class LibExport PVMappingFilterString4Bsort: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::nraw_table_line const& values);

protected:
	static float compute_str_factor(QString const& str);

	CLASS_FILTER(PVMappingFilterString4Bsort)
};

}

#endif	/* PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H */
