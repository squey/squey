/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVWax.h>
#include "massive_common.h"

/*****************************************************************************
 * PVHive::PVHive::call_object specialization
 *****************************************************************************/

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Block::set_prop, b, args)
{
	call_object_default<Block, FUNC(Block::set_prop)>(b, args);
	refresh_observers(&(b->get_prop(std::get<0>(args))));
}

PVHIVE_CALL_OBJECT_BLOCK_END()

/*****************************************************************************
 * property accessor
 *****************************************************************************/

Property* get_prop(Block& b, int i)
{
	return &b.get_prop(i);
}

/*****************************************************************************
 * ::action()
 *****************************************************************************/

void BlockAct::action()
{
	PVACTOR_CALL(*this, &Block::set_value, _value);
}


void PropertyAct::action()
{
	PVACTOR_CALL(*this, &Block::set_prop, _index, Property(_value));
}
