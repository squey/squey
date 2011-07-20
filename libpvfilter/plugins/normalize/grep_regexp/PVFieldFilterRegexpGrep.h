//! \file PVFieldFilterGrep.h
//! $Id: PVFieldFilterRegexpGrep.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDFILTERREGEXPGREP_H
#define PVFILTER_PVFIELDFILTERREGEXPGREP_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilter.h>
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
