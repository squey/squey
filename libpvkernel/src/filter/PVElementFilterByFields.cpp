/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
PVCore::PVElement& PVFilter::PVElementFilterByFields::operator()(PVCore::PVElement& elt)
{
	elt.fields() = _ff(elt.fields());
	return elt;
}
