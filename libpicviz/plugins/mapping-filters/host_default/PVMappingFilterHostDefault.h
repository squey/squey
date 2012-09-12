/**
 * \file PVMappingFilterHostDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

class PVMappingFilterHostDefault: public PVMappingFilter
{
public:
	PVMappingFilterHostDefault(PVCore::PVArgumentList const& args = PVMappingFilterHostDefault::default_args());

public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	QString get_human_name() const override { return QString("Default"); }
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::DecimalType get_decimal_type() const { return PVCore::UnsignedIntegerType; }

private:
	bool _case_sensitive;
	
	CLASS_FILTER(PVMappingFilterHostDefault)
};

}

#endif
