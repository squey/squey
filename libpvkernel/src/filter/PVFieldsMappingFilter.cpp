/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>
#include <list>

/******************************************************************************
 *
 * PVFilter::PVCore::PVFieldsMappingFilter::PVCore::PVFieldsMappingFilter
 *
 *****************************************************************************/
PVFilter::PVFieldsMappingFilter::PVFieldsMappingFilter(size_t idx, PVFieldsBaseFilter_f func)
    : _idx(idx), _func(func)
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
	PVCore::list_fields::iterator it_curf = fields.begin();
	std::advance(it_curf, _idx);
	tmp_fields.push_back(*it_curf);
	fields.erase(it_curf);

	// Apply the filter
	PVCore::list_fields& final_fields = _func(tmp_fields);

	// Move generated field in the list.
	PVCore::list_fields::iterator itins = fields.begin();
	std::advance(itins, _idx);
	fields.splice(itins, final_fields);

	return fields;
}
