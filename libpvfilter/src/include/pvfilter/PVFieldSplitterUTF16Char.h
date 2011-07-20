//! \file PVFieldSplitterUTF16Char.h
//! $Id: PVFieldSplitterUTF16Char.h 3165 2011-06-16 05:05:39Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFIELDSPLITTERUTF16CHAR_H
#define PVFILTER_PVFIELDSPLITTERUTF16CHAR_H

#include <pvcore/general.h>
#include <pvcore/PVField.h>
#include <pvfilter/PVFieldsFilter.h>
#include <QChar>

namespace PVFilter {

class LibFilterDecl PVFieldSplitterUTF16Char : public PVFieldsFilter<one_to_many> {
public:
	PVFieldSplitterUTF16Char(PVCore::PVArgumentList const& args);
public:
	virtual void set_args(PVCore::PVArgumentList const& args);
protected:
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);
protected:
	QChar _c;

	CLASS_FILTER(PVFilter::PVFieldSplitterUTF16Char)
};

}

#endif
