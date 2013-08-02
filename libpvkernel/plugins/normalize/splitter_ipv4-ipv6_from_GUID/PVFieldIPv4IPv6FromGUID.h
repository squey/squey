/**
 * \file PVFieldIPv4IPv6FromGUID.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVFILTER_PVFIELDIPV4IPV6FROMGUID_H
#define PVFILTER_PVFIELDIPV4IPV6FROMGUID_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldIPv4IPv6FromGUID : public PVFieldsFilter<one_to_many> {

public:
	PVFieldIPv4IPv6FromGUID(PVCore::PVArgumentList const& args = PVFieldIPv4IPv6FromGUID::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

private:
	bool _ipv4;
	bool _ipv6;

	CLASS_FILTER(PVFilter::PVFieldIPv4IPv6FromGUID)
};

}

#endif // PVFILTER_PVFIELDIPV4IPV6FROMGUID_H
