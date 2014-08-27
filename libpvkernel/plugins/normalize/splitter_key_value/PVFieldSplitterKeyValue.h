/**
 * \file PVFieldSplitterKeyValue.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDSPLITTERKEYVALUE_H
#define PVFILTER_PVFIELDSPLITTERKEYVALUE_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldSplitterKeyValue : public PVFieldsSplitter {

public:
	PVFieldSplitterKeyValue(PVCore::PVArgumentList const& args = PVFieldSplitterKeyValue::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field) override;

private:

	CLASS_FILTER(PVFilter::PVFieldSplitterKeyValue)
};

}

#endif // PVFILTER_PVFIELDSPLITTERKEYVALUE_H
