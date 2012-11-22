/**
 * \file PVFieldDuplicate.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include "PVFieldDuplicate.h"

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::PVFieldDuplicate
 *
 *****************************************************************************/
PVFilter::PVFieldDuplicate::PVFieldDuplicate(PVCore::PVArgumentList const& args) :
	PVFieldsFilter<PVFilter::one_to_many>()
{
	INIT_FILTER(PVFilter::PVFieldDuplicate, args);
}

void PVFilter::PVFieldDuplicate::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_n = std::max((uint32_t) args["n"].toUInt(), (uint32_t) 2);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldDuplicate)
{
	PVCore::PVArgumentList args;
	args["n"] = 2;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldDuplicate::one_to_many
 *
 *****************************************************************************/
PVCore::list_fields::size_type PVFilter::PVFieldDuplicate::one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field)
{
	PVCore::list_fields::size_type ret = 0;

	for (size_t i = 0; i < _n; i++) {
		l.insert(it_ins, field);
		ret++;
	}

	return ret;
}


IMPL_FILTER(PVFilter::PVFieldDuplicate)
