/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

/**
 * Host mapping work on string but separate Ip from String.
 * 
 * Ip are in 0 to 2**31 and string are in 2*µ31 to 2**32.
 */
class PVMappingFilterHostDefault: public PVMappingFilter
{
public:
	PVMappingFilterHostDefault();

public:
	/**
	 * This mapping apply cell by cell.
	 *
	 * It compute mapping value.
	 */
	Inendi::PVMappingFilter::decimal_storage_type process_cell(const char* buf, size_t size) override;

	/**
	 * Meta information from this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterHostDefault)
};

}

#endif
