//! \file PVCore::PVFieldSplitterUTF16Char.h
//! $Id: PVFieldSplitterURL.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSPLITTERURL_H
#define PVFILTER_PVFIELDSPLITTERURL_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <QChar>

namespace PVFilter {

class PVFieldSplitterURL : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterURL();
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
protected:
	CLASS_FILTER(PVFilter::PVFieldSplitterURL)

protected:

};

}

#endif
