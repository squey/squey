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
PVFilter::PVFieldConverterStruct::PVFieldConverterStruct() : PVFieldsConverter()
{
	INIT_FILTER_NOPARAM(PVFilter::PVFieldConverterStruct);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterStruct::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldConverterStruct::one_to_one(PVCore::PVField& field)
{
	field.set_end(
	    std::remove_if(field.begin(), field.end(), [](char c) { return std::isalnum(c); }));

	return field;
}

IMPL_FILTER_NOPARAM(PVFilter::PVFieldConverterStruct)
