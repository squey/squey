/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERIPV4DEFAULT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

struct ipv4_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterIPv4Default: public PVMappingFilter
{
public:
	Inendi::PVMappingFilter::decimal_storage_type process_cell(const char* buf, size_t size) override
	{
		return ipv4_mapping::process_utf8(buf, size, this);
	}
	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	CLASS_FILTER(PVMappingFilterIPv4Default)
};

}

#endif
