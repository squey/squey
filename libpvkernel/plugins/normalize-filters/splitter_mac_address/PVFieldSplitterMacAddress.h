/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVFIELDMACADDRESS_H
#define PVFILTER_PVFIELDMACADDRESS_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

/**
 * Split mac address with device part and constructor part.
 */
class PVFieldSplitterMacAddress : public PVFieldsFilter<one_to_many>
{

  public:
	PVFieldSplitterMacAddress();
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields& l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField& field);

	CLASS_FILTER_NOPARAM(PVFilter::PVFieldSplitterMacAddress)
};
}

#endif // PVFILTER_PVFIELDMACADDRESS_H
