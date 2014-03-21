/**
 * \file PVFieldSplitterDnsFqdn.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDSPLITTERDNSFQDN_H
#define PVFILTER_PVFIELDSPLITTERDNSFQDN_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterDnsFqdn : public PVFieldsFilter<one_to_many> {

public:
	static const char* N;
	static const char* TLD1;
	static const char* TLD2;
	static const char* TLD3;
	static const char* SUBD1;
	static const char* SUBD2;
	static const char* SUBD3;
	static const char* SUBD1_REV;
	static const char* SUBD2_REV;
	static const char* SUBD3_REV;

public:
	PVFieldSplitterDnsFqdn(PVCore::PVArgumentList const& args = PVFieldSplitterDnsFqdn::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

private:
	int  _n;
	bool _tld1;
	bool _tld2;
	bool _tld3;
	bool _subd1;
	bool _subd2;
	bool _subd3;
	bool _subd1_rev;
	bool _subd2_rev;
	bool _subd3_rev;
	bool _need_rev;

	CLASS_FILTER(PVFilter::PVFieldSplitterDnsFqdn)
};

}

#endif // PVFILTER_PVFIELDSPLITTERDNSFQDN_H
