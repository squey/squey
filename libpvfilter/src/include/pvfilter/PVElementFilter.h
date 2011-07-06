//! \file PVElementFilter.h
//! $Id: PVElementFilter.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVELEMENTFILTER_H
#define PVFILTER_PVELEMENTFILTER_H

#include <pvcore/general.h>
#include <pvfilter/PVFilterFunction.h>
#include <pvcore/PVElement.h>

namespace PVFilter {

class PVElementFilter: public PVFilterFunction<PVCore::PVElement, PVElementFilter>
{
public:
	typedef PVElementFilter FilterT;
	typedef boost::shared_ptr<PVElementFilter> p_type;

public:
	virtual PVCore::PVElement& operator()(PVCore::PVElement& in) = 0;
};

typedef PVElementFilter::func_type PVElementFilter_f;

}

#endif
