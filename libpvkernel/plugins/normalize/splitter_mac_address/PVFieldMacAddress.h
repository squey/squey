/**
 * \file PVFieldMacAddress.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDMACADDRESS_H
#define PVFILTER_PVFIELDMACADDRESS_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldMacAddress : public PVFieldsFilter<one_to_many>
{
public:
	static const char* UPPERCASE;

public:
	PVFieldMacAddress(PVCore::PVArgumentList const& args = PVFieldMacAddress::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);

	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l,
	                                           PVCore::list_fields::iterator it_ins,
	                                           PVCore::PVField &field);

private:
	bool _uppercased;

	CLASS_FILTER(PVFilter::PVFieldMacAddress)
};

}

#endif // PVFILTER_PVFIELDMACADDRESS_H
