/**
 * \file PVFieldConverterValueMapper.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldConverterValueMapper.h"

/******************************************************************************
 *
 * PVFilter::PVFieldGUIDToIP::PVFieldGUIDToIP
 *
 *****************************************************************************/
PVFilter::PVFieldConverterValueMapper::PVFieldConverterValueMapper(PVCore::PVArgumentList const& args) :
	PVFieldsConverter()
{
	INIT_FILTER(PVFilter::PVFieldConverterValueMapper, args);
}

void PVFilter::PVFieldConverterValueMapper::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldConverterValueMapper)
{
	PVCore::PVArgumentList args;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterValueMapper::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterValueMapper::one_to_one(PVCore::PVField& field)
{
}


IMPL_FILTER(PVFilter::PVFieldConverterValueMapper)
