/**
 * \file PVMappingFilterIntegerOct.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGEROCT_H
#define PVFILTER_PVMAPPINGFILTERINTEGEROCT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

namespace Picviz {

class PVMappingFilterIntegerOct;

struct integer_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterIntegerOct: public PVPureMappingFilter<integer_mapping>
{
	friend class integer_mapping;

public:
	PVMappingFilterIntegerOct(PVCore::PVArgumentList const& args = PVMappingFilterIntegerOct::default_args());

public:
	QString get_human_name() const override { return QString("Octal"); }
	PVCore::DecimalType get_decimal_type() const override;
	void set_args(PVCore::PVArgumentList const& args) override;

protected:
	inline bool is_signed() const { return _signed; }

private:
	bool _signed;

	CLASS_FILTER(PVMappingFilterIntegerOct)
};

}

#endif
