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

#include <pvkernel/filter/PVFieldsFilter.h> // for PVFieldsFilter, etc

#include <pvkernel/core/PVField.h>   // for PVField
#include <pvkernel/core/PVElement.h> // for list_fields, PVElement

#include <QString> // for QString

#include <cstddef>  // for size_t
#include <iterator> // for advance, distance
#include <list>     // for _List_iterator, etc

// The namespace has to be specified for template function specialisation
namespace PVFilter
{

// Template specialisation for generic L<F>& operator()(L<F>&)
template <>
PVCore::list_fields& PVFieldsFilter<one_to_one>::operator()(PVCore::list_fields& fields)
{
	PVCore::list_fields::iterator it, ite;
	ite = fields.end();
	for (it = fields.begin(); it != ite; it++) {
		PVCore::PVField& ret = this->one_to_one(*it);
		if (!ret.valid()) {
			ret.elt_parent()->set_invalid();
			break;
		}
		if (ret.filtered()) {
			ret.elt_parent()->set_filtered();
			break;
		}
		*it = ret;
	}
	return fields;
};

template <>
PVCore::list_fields& PVFieldsFilter<one_to_many>::operator()(PVCore::list_fields& fields)
{
	PVCore::list_fields::iterator it, ite, it_cur;
	ite = fields.end();
	it = fields.begin();
	while (it != ite) {
		it_cur = it;
		it++;
		size_t field_count = this->one_to_many(fields, it_cur, *it_cur);
		if ((_fields_expected != 0 and field_count != _fields_expected) or field_count == 0) {
			// The splitting didn't work
			// Invalidate the parent
			it_cur->elt_parent()->set_invalid();

			// Get the iterator range over all created but invalid elements
			size_t pos = std::distance(fields.begin(), it_cur);
			std::advance(it_cur, 1);
			auto begin_it = fields.begin();
			std::advance(begin_it, pos - field_count);

			// Mark all these fields as invalid
			for (auto it = begin_it; it != it_cur; it++) {
				it->set_invalid();
			}

			// And remove them
			fields.erase(begin_it, it_cur);
			break;
		}
		fields.erase(it_cur);
	}
	return fields;
};

template <>
PVCore::list_fields& PVFieldsFilter<many_to_many>::operator()(PVCore::list_fields& fields)
{
	return many_to_many(fields);
};

template <>
QString PVFieldsFilter<one_to_one>::type_name()
{
	return {"converter"};
}

template <>
QString PVFieldsFilter<one_to_many>::type_name()
{
	return {"splitter"};
}

template <>
QString PVFieldsFilter<many_to_many>::type_name()
{
	return {"generic"};
}
} // namespace PVFilter

// Explicit template instanciation is needed for clang to avoid creating multiple
// instances of the same singleton type
template class PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_one>>;
template class PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::one_to_many>>;
template class PVCore::PVClassLibrary<PVFilter::PVFieldsFilter<PVFilter::many_to_many>>;