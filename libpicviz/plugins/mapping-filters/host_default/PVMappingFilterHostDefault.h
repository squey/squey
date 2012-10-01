/**
 * \file PVMappingFilterHostDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

struct host_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterHostDefault: public PVPureMappingFilter<host_mapping>
{
public:
	PVMappingFilterHostDefault(PVCore::PVArgumentList const& args = PVMappingFilterHostDefault::default_args());

public:
	QString get_human_name() const override { return QString("Default"); }
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

private:
	bool _case_sensitive;
	
	CLASS_FILTER(PVMappingFilterHostDefault)
};

}

#endif
