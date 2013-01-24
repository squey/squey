/**
 * \file PVMappingFilterIntegerHex.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGERHEX_H
#define PVFILTER_PVMAPPINGFILTERINTEGERHEX_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

namespace Picviz {

class PVMappingFilterIntegerHex;

struct integer_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterIntegerHex: public PVPureMappingFilter<integer_mapping>
{
	friend class integer_mapping;

public:
	PVMappingFilterIntegerHex(PVCore::PVArgumentList const& args = PVMappingFilterIntegerHex::default_args());

public:
	QString get_human_name() const override { return QString("Hexadecimal"); }
	PVCore::DecimalType get_decimal_type() const override;
	void set_args(PVCore::PVArgumentList const& args) override;

protected:
	inline bool is_signed() const { return _signed; }

private:
	bool _signed;

	CLASS_FILTER(PVMappingFilterIntegerHex)
};

}

#endif
