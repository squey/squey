/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsBaseFilter_p
#include <pvkernel/filter/PVFieldsMappingFilter.h>

#include <algorithm> // for move
#include <cassert>   // for assert
#include <cstddef>   // for size_t

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldsMappingFilter::PVCore::PVFieldsMappingFilter
 *
 *****************************************************************************/
PVFilter::PVFieldsMappingFilter::PVFieldsMappingFilter(size_t idx, PVFieldsBaseFilter_p func)
    : _idx(idx), _func(std::move(func))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldsMappingFilter::operator
 *
 *****************************************************************************/
PVCore::list_fields& PVFilter::PVFieldsMappingFilter::many_to_many(PVCore::list_fields& fields)
{
	// TODO: this *can* be optimised !
	if (fields.size() == 0)
		return fields;

	assert(_idx < fields.size());

	// Create list of input field for the filter
	PVCore::list_fields tmp_fields;
	auto it_curf = fields.begin();
	std::advance(it_curf, _idx);
	tmp_fields.push_back(*it_curf);
	fields.erase(it_curf);

	// Apply the filter
	PVCore::list_fields& final_fields = (*_func)(tmp_fields);

	// If any fields is incorrect, abort the splitting on this element.
	if (std::any_of(final_fields.begin(), final_fields.end(),
	                [](PVCore::PVField& f) { return f.filtered() or not f.valid(); })) {
		fields.clear();
		return fields;
	}

	// Move generated field in the list.
	auto itins = fields.begin();
	std::advance(itins, _idx);
	fields.splice(itins, final_fields);

	return fields;
}
