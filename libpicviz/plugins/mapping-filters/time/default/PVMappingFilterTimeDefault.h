/**
 * \file PVMappingFilterTimeDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERTIMEDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

#include <unicode/calendar.h>

namespace Picviz {

class PVMappingFilterTimeDefault: public PVMappingFilter
{
public:
	PVMappingFilterTimeDefault(PVCore::PVArgumentList const& args = PVMappingFilterTimeDefault::default_args());

public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }
protected:
	virtual int32_t cal_to_int(Calendar* cal, bool& success);

	CLASS_FILTER(PVMappingFilterTimeDefault)
};

}

#endif
