//! \file PVFieldsFilter.cpp
//! $Id: PVFieldsFilter.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/PVField.h>
#include <pvfilter/PVFieldsFilter.h>

// The namespace has to be specified for template function specialisation
namespace PVFilter {

// Template specialisation for generic L<F>& operator()(L<F>&)
template<>
PVCore::list_fields& PVFieldsFilter<one_to_one>::operator()(PVCore::list_fields& fields)
{
	PVCore::list_fields::iterator it,ite;
	ite = fields.end();
	for (it = fields.begin(); it != ite; it++) {
		PVCore::PVField& ret = this->one_to_one(*it);
		if (!ret.valid()) {
			ret.elt_parent()->set_invalid();
			break;
		}
		*it = ret;
	}
	return fields;
};

template<>
PVCore::list_fields& PVFieldsFilter<one_to_many>::operator()(PVCore::list_fields& fields)
{
	PVCore::list_fields::iterator it,ite,it_cur;
	ite = fields.end();
	it = fields.begin();
	while (it != ite) {
		it_cur = it;
		it++;
		if (this->one_to_many(fields, it_cur, *it_cur) == 0) {
			(*it_cur).elt_parent()->set_invalid();
			fields.erase(it_cur);
			break;
		}
		fields.erase(it_cur);
	}
	return fields;
};


template<>
PVCore::list_fields& PVFieldsFilter<many_to_many>::operator()(PVCore::list_fields& fields)
{
	return fields;
};

}
