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

#include <pvkernel/filter/PVElementFilterByAxes.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <cstddef> // for size_t
#include <list>    // for _List_iterator, list

#include "pvkernel/core/PVElement.h"
#include "pvkernel/filter/PVFilterFunction.h"

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
