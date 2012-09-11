/**
 * \file PVMappingFilterString4Bsort.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <picviz/PVMappingFilter.h>

#include <tbb/atomic.h>

namespace Picviz {

class PVMappingFilterString4Bsort: public PVMappingFilter
{
public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	QString get_human_name() const { return QString("Pseudo-sort on the first 4 bytes"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

protected:
	CLASS_FILTER(PVMappingFilterString4Bsort)
};

}

#endif	/* PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H */
