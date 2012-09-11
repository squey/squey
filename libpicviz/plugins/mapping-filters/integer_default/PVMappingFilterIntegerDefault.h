/**
 * \file PVMappingFilterIntegerDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGER_H
#define PVFILTER_PVMAPPINGFILTERINTEGER_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterIntegerDefault: public PVMappingFilter
{
public:
	PVMappingFilterIntegerDefault(PVCore::PVArgumentList const& args = PVMappingFilterIntegerDefault::default_args());

public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	QString get_human_name() const override { return QString("default"); }
	PVCore::DecimalType get_decimal_type() const override;
	void set_args(PVCore::PVArgumentList const& args) override;

private:
	bool _signed;

	CLASS_FILTER(PVMappingFilterIntegerDefault)
};

}

#endif
