/**
 * \file PVFieldSplitterUTF16Char.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVFieldSplitterUTF16Char.h>
#include <pvkernel/core/PVBufferSlice.h>

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterUTF16Char::PVCore::PVFieldSplitterUTF16Char
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterUTF16Char::PVFieldSplitterUTF16Char(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterUTF16Char, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterUTF16Char)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterUTF16Char)
{
	PVCore::PVArgumentList args;
	args["c"] = QVariant(QChar(' '));
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterUTF16Char::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterUTF16Char::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_c = args["c"].toChar();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterUTF16Char::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterUTF16Char::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	return field.split_qchar<PVCore::list_fields>(l, _c, it_ins);
}

IMPL_FILTER(PVFilter::PVFieldSplitterUTF16Char)
