/**
 * \file PVMappingFilterStringDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERSTRINGDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>
#include <tbb/atomic.h>

#include <pvkernel/core/stdint.h>

namespace Picviz {

struct string_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterStringDefault: public PVPureMappingFilter<string_mapping>
{
	friend class string_mapping;
public:
	PVMappingFilterStringDefault(PVCore::PVArgumentList const& args = PVMappingFilterStringDefault::default_args());

public:
	// Overloaded from PVFunctionArgs::set_args
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }
	QString get_human_name() const override { return "Default"; }

protected:
	bool case_sensitive() const { return _case_sensitive; }

private:
	bool _case_sensitive;
	CLASS_FILTER(PVMappingFilterStringDefault)
};

}

#endif
