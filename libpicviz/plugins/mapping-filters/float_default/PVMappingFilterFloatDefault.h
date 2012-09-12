/**
 * \file PVMappingFilterFloatDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>

namespace Picviz {

class PVMappingFilterFloatDefault: public PVMappingFilter
{
public:
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::FloatType; }

	CLASS_FILTER(PVMappingFilterFloatDefault)
};

}

#endif
