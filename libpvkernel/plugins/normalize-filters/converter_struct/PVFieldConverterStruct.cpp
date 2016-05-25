/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldConverterStruct.h"

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStruct::PVFieldConverterStruct
 *
 *****************************************************************************/
PVFilter::PVFieldConverterStruct::PVFieldConverterStruct(PVCore::PVArgumentList const& args)
    : PVFieldsConverter()
{
	INIT_FILTER(PVFilter::PVFieldConverterStruct, args);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStruct::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterStruct::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
}

DEFAULT_ARGS_FILTER(PVFilter::PVFieldConverterStruct)
{
	PVCore::PVArgumentList args;

	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStruct::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterStruct::one_to_one(PVCore::PVField& field)
{

	std::string str(field.begin(), field.size());
	str.resize(std::distance(str.begin(), std::remove_if(str.begin(), str.end(),
	                                                     [](char c) { return std::isalnum(c); })));

	field.allocate_new(str.size());
	field.set_end(std::copy(str.begin(), str.end(), field.begin()));

	return field;
}

IMPL_FILTER(PVFilter::PVFieldConverterStruct)
