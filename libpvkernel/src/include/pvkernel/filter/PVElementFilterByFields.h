//! \file PVElementFilterByFields.h
//! $Id: PVElementFilterByFields.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

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
};

}

#endif
