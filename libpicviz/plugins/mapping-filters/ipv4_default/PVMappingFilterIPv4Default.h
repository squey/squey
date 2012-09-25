/**
 * \file PVMappingFilterIPv4Default.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVPureMappingFilter.h>

namespace Picviz {

struct ipv4_mapping
{
	static Picviz::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Picviz::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterIPv4Default: public PVPureMappingFilter<ipv4_mapping>
{
public:
	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	CLASS_FILTER(PVMappingFilterIPv4Default)
};

}

#endif
