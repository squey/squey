/**
 * \file PVElementFilterByFields.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVELEMENTFILTERBYFIELDS_H
#define PVFILTER_PVELEMENTFILTERBYFIELDS_H

#include <pvkernel/core/general.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/core/PVElement.h>

namespace PVFilter {

class LibKernelDecl PVElementFilterByFields : public PVElementFilter {
public:
	PVElementFilterByFields(PVFieldsBaseFilter_f fields_f);
public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);
protected:
	PVFieldsBaseFilter_f _ff;

	CLASS_FILTER_NONREG_NOPARAM(PVElementFilterByFields)
};

}

#endif
