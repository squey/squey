//! \file PVElementFilterByFields.cpp
//! $Id: PVElementFilterByFields.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

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

IMPL_FILTER_NOPARAM(PVFilter::PVElementFilterByFields)
