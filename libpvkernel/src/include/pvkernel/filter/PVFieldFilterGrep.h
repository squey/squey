/**
 * \file PVFieldFilterGrep.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVFIELDFILTERGREP_H
#define PVFILTER_PVFIELDFILTERGREP_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <QString>

namespace PVFilter {

class LibKernelDecl PVFieldFilterGrep: public PVFieldsFilter<one_to_one> {
public:
	PVFieldFilterGrep(PVCore::PVArgumentList const& args = PVFieldFilterGrep::default_args());
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
public:
	virtual PVCore::PVField& one_to_one(PVCore::PVField& obj);
protected:
	QString _str;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVFieldFilterGrep)
};

}

#endif
