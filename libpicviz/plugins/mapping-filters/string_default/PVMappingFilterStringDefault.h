/**
 * \file PVMappingFilterStringDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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
	// Overloaded from PVFunctionArgs::set_args
	void set_args(PVCore::PVArgumentList const& args);
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

private:
	bool _case_sensitive;
	CLASS_FILTER(PVMappingFilterStringDefault)
};

}

#endif
