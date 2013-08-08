/**
 * \file PVFieldSplitterIP.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef PVFILTER_PVFIELDSPLITTERIP_H
#define PVFILTER_PVFIELDSPLITTERIP_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterIP : public PVFieldsSplitter {

public:
	static const QString sep;

public:
	PVFieldSplitterIP(PVCore::PVArgumentList const& args = PVFieldSplitterIP::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

private:
	bool _ipv6;
	QString _params;
	std::vector<size_t> _indexes;

	CLASS_FILTER(PVFilter::PVFieldSplitterIP)
};

}

#endif // PVFILTER_PVFIELDSPLITTERIP_H
