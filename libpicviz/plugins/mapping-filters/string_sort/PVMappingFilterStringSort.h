/**
 * \file PVMappingFilterStringSort.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRINGSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRINGSORT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

class PVMappingFilterStringSort: public PVMappingFilter
{
public:
	PVMappingFilterStringSort(PVCore::PVArgumentList const& args = PVMappingFilterStringSort::default_args());
public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	QString get_human_name() const override { return QString("Sort"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }
	
	CLASS_FILTER(PVMappingFilterStringSort)
};

}

#endif
