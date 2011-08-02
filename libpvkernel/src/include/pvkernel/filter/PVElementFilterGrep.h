//! \file PVElementFilterGrep.h
//! $Id: PVElementFilterGrep.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVELEMENTFILTERGREP_H
#define PVFILTER_PVELEMENTFILTERGREP_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <QString>

namespace PVFilter {

class LibKernelDecl PVElementFilterGrep: public PVElementFilter {
public:
	PVElementFilterGrep(PVCore::PVArgumentList const& args = PVElementFilterGrep::default_args());
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
public:
	virtual PVCore::PVElement& operator()(PVCore::PVElement& obj);
protected:
	QString _str;
	bool _inverse;

	CLASS_FILTER(PVFilter::PVElementFilterGrep)
};

}

#endif
