//! \file PVElementFilterByFields.h
//! $Id: PVElementFilterByFields.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVELEMENTFILTERBYFIELDS_H
#define PVFILTER_PVELEMENTFILTERBYFIELDS_H

#include <pvcore/general.h>
#include <pvfilter/PVElementFilter.h>
#include <pvfilter/PVFilterFunction.h>
#include <pvfilter/PVFieldsFilter.h>
#include <pvcore/PVElement.h>

namespace PVFilter {

class LibExport PVElementFilterByFields : public PVElementFilter {
public:
	PVElementFilterByFields(PVFieldsBaseFilter_f fields_f);
public:
	PVCore::PVElement& operator()(PVCore::PVElement& elt);
protected:
	PVFieldsBaseFilter_f _ff;

	CLASS_FILTER(PVFilter::PVElementFilterByFields)
};

}

#endif
