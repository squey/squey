/**
 * \file functional_objs.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVCallHelper.h>
#include "functional_objs.h"


PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

template <>
void PVHive::PVHive::call_object<FUNC(PropertyEntity::set_prop)>(PropertyEntity* e, PVCore::PVTypeTraits::function_traits<decltype(&PropertyEntity::set_prop)>::arguments_type const& args)
{
	std::cout << "CALL ::set_prop on PE " << e->get_prop() << std::endl;

	call_object_default<PropertyEntity, FUNC(PropertyEntity::set_prop)>(e, args);
	refresh_observers(e->get_prop());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
