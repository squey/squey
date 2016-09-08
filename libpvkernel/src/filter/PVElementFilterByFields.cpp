/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVElementFilterByFields.h>

#include <pvkernel/core/PVElement.h> // for PVElement

/******************************************************************************
 *
 * PVFilter::PVElementFilterByFields::operator()
 *
 *****************************************************************************/
PVCore::PVElement& PVFilter::PVElementFilterByFields::operator()(PVCore::PVElement& elt)
{
	for (auto f : _ff) {
		elt.fields() = (*f)(elt.fields());
	}
	return elt;
}
