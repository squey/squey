/**
 * \file massive_common.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
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

const Property& get_prop(const Block& b, int i)
{
	return b.get_prop(i);
}

/*****************************************************************************
 * display stats
 *****************************************************************************/

void print_stat(const char *what, tbb::tick_count t1, tbb::tick_count t2, long num)
{
	double dt = (t2 - t1).seconds();

	std::cout << num << " " << what << " in " << dt
	          << " sec => ops per sec: " << num / dt << std::endl;

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
