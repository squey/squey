/**
 * \file PVFieldSplitterKeyValue.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterKeyValue.h"

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValue::PVFieldSplitterKeyValue(PVCore::PVArgumentList const& args) :
	PVFieldsSplitter()
{
	INIT_FILTER(PVFilter::PVFieldSplitterKeyValue, args);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterKeyValue::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldSplitterKeyValue)
{
	PVCore::PVArgumentList args;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValue::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldSplitterKeyValue::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;

	return ret;
}
