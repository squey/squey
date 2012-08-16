/**
 * \file functional_objs.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "functional_objs.h"


PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

template <>                                                                                                   // boost::reference_wrapper<Property const>
void PVHive::PVHive::call_object<PropertyEntity, decltype(&PropertyEntity::set_prop), &PropertyEntity::set_prop, Property>(PropertyEntity* e, Property&& p)
{
	std::cout << "CALL ::set_prop on PE " << e->get_prop() << std::endl;

	call_object_default<PropertyEntity, decltype(&PropertyEntity::set_prop), &PropertyEntity::set_prop>(e, std::forward<Property>(p));
	refresh_observers(e->get_prop());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
