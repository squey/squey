/**
 * \file PVElementFilterByFields.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVElementFilterByFields.h>


/******************************************************************************
 *
 * PVFilter::PVElementFilterByFields::PVElementFilterByFields
 *
 *****************************************************************************/
PVFilter::PVElementFilterByFields::PVElementFilterByFields(PVFieldsBaseFilter_f fields_f)
	: PVElementFilter()
{
	_ff = fields_f;
	INIT_FILTER_NOPARAM(PVFilter::PVElementFilterByFields);
}

/******************************************************************************
 *
 * PVFilter::PVElementFilterByFields::operator()
 *
 *****************************************************************************/
PVCore::PVElement& PVFilter::PVElementFilterByFields::operator()(PVCore::PVElement &elt)
{
	elt.fields() = _ff(elt.fields());
	return elt;
}
