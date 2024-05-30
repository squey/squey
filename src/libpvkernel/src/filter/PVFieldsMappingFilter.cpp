//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
