/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

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
		// FIXME : _fields_expected is amnost not use. It should always be set for better invalid
		// element detection.
		if ((_fields_expected != 0 and field_count != _fields_expected) or field_count == 0) {
			it_cur->set_invalid();
			(*it_cur).elt_parent()->set_invalid();
			fields.erase(it_cur);
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
	return QString("converter");
}

template <>
QString PVFieldsFilter<one_to_many>::type_name()
{
	return QString("splitter");
}

template <>
QString PVFieldsFilter<many_to_many>::type_name()
{
	return QString("generic");
}
}
