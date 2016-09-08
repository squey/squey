/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <pvkernel/filter/PVElementFilterByAxes.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <pvkernel/core/PVArgument.h> // for PVArgumentList

#include <cstddef> // for size_t
#include <list>    // for _List_iterator, list

/******************************************************************************
 * PVFilter::PVElementFilterByAxes::PVElementFilterByAxes
 *****************************************************************************/

PVFilter::PVElementFilterByAxes::PVElementFilterByAxes(const fields_mask_t& fields_mask)
    : PVFilter::PVElementFilterByFields(), _fields_mask(fields_mask)
{
	INIT_FILTER_NOPARAM(PVFilter::PVElementFilterByAxes);
}

/******************************************************************************
 * PVFilter::PVElementFilterByAxes::operator()
 *****************************************************************************/

PVCore::PVElement& PVFilter::PVElementFilterByAxes::operator()(PVCore::PVElement& elt)
{
	elt = PVElementFilterByFields::operator()(elt);

	size_t i = 0;
	auto& fields = elt.fields();

	auto it = fields.begin();

	while (it != fields.end()) {
		if (!_fields_mask[i]) {
			it = fields.erase(it);
		} else {
			++it;
		}
		++i;
	}

	return elt;
}
