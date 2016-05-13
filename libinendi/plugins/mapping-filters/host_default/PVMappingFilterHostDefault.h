/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Host mapping work on string but separate Ip from String.
 *
 * Ip are in 0 to 2**31 and string are in 2*µ31 to 2**32.
 */
class PVMappingFilterHostDefault : public PVMappingFilter
{
  public:
	PVMappingFilterHostDefault();

  public:
	PVMappingFilter::decimal_storage_type* operator()(PVCol const col,
	                                                  PVRush::PVNraw const& nraw) override;

	/**
	 * Meta information from this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	CLASS_FILTER_NOPARAM(PVMappingFilterHostDefault)
};
}

#endif
