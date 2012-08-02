/**
 * \file PVFieldFilterRegexpGrep.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVFIELDFILTERREGEXPGREP_H
#define PVFILTER_PVFIELDFILTERREGEXPGREP_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <QRegExp>

namespace PVFilter {

class PVFieldFilterRegexpGrep: public PVFieldsFilter<one_to_one> {
public:
	PVFieldFilterRegexpGrep(PVCore::PVArgumentList const& args = PVFieldFilterRegexpGrep::default_args());
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
public:
	virtual PVCore::PVField& one_to_one(PVCore::PVField& obj);
protected:
	QRegExp _rx;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
};

}

#endif
