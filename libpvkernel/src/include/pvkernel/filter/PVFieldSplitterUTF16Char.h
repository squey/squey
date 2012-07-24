/**
 * \file PVFieldSplitterUTF16Char.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVFIELDSPLITTERUTF16CHAR_H
#define PVFILTER_PVFIELDSPLITTERUTF16CHAR_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <QChar>

namespace PVFilter {

class LibKernelDecl PVFieldSplitterUTF16Char : public PVFieldsFilter<one_to_many> {
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
