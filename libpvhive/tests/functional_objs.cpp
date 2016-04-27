/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVWax.h>
#include "functional_objs.h"

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(PropertyEntity::set_prop, e, args)
{
	std::cout << "CALL ::set_prop on PE " << e->get_prop() << std::endl;

	call_object_default<PropertyEntity, FUNC(PropertyEntity::set_prop)>(e, args);
	refresh_observers(e->get_prop());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
