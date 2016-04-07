/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDMACADDRESS_H
#define PVFILTER_PVFIELDMACADDRESS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterMacAddress : public PVFieldsFilter<one_to_many>
{
public:
	static const char* UPPERCASE;

public:
	PVFieldSplitterMacAddress(PVCore::PVArgumentList const& args = PVFieldSplitterMacAddress::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);

	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField &field);

private:
	bool _uppercased; //!< Wether field should be uppercase. FIXME : Doesn't work.

	CLASS_FILTER(PVFilter::PVFieldSplitterMacAddress)
};

}

#endif // PVFILTER_PVFIELDMACADDRESS_H
