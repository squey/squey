/**
 * \file PVFieldSplitterChar.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/filter/PVFieldSplitterChar.h>
#include <pvkernel/core/PVBufferSlice.h>

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldSplitterChar::PVCore::PVFieldSplitterChar
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterChar::PVFieldSplitterChar(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldSplitterChar, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterChar)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterChar)
{
	PVCore::PVArgumentList args;
	args["c"] = QVariant(' ');
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterChar::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterChar::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_c = args["c"].toChar().toLatin1();
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterChar::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterChar::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	return field.split<PVCore::list_fields>(l, _c, it_ins);
}

IMPL_FILTER(PVFilter::PVFieldSplitterChar)
