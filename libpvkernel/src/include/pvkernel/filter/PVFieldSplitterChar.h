//! \file PVFieldSplitterChar.h
//! $Id: PVFieldSplitterChar.h 3165 2011-06-16 05:05:39Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSPLITTERCHAR_H
#define PVFILTER_PVFIELDSPLITTERCHAR_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class LibKernelDecl PVFieldSplitterChar : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterChar(PVCore::PVArgumentList const& args = PVFieldSplitterChar::default_args());
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
protected:
	char _c;

	CLASS_FILTER(PVFilter::PVFieldSplitterChar)
};

}

#endif
