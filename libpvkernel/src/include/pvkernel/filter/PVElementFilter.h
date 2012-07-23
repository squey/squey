/**
 * \file PVElementFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVELEMENTFILTER_H
#define PVFILTER_PVELEMENTFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVElement.h>

namespace PVFilter {

class PVElementFilter: public PVFilterFunctionBase<PVCore::PVElement&, PVCore::PVElement&>
{
public:
	typedef PVElementFilter FilterT;
	typedef boost::shared_ptr<PVElementFilter> p_type;

public:
	PVCore::PVElement& operator()(PVCore::PVElement& in) { return in; }

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilter)
};

typedef PVElementFilter::func_type PVElementFilter_f;

}

#endif
