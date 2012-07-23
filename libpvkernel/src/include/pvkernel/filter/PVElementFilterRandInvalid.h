/**
 * \file PVElementFilterRandInvalid.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVELEMENTFILTERRANDINVALID_H
#define PVFILTER_PVELEMENTFILTERRANDINVALID_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>

namespace PVFilter {

// This class will randomly invalidate elements
// This is used for the controller's test cases
class LibKernelDecl PVElementFilterRandInvalid : public PVElementFilter {
public:
	PVElementFilterRandInvalid();
public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterRandInvalid)
};

}

#endif
