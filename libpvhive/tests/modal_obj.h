/**
 * \file modal_obj.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef MODAL_OBJ_H
#define MODAL_OBJ_H

#include <pvkernel/core/PVSharedPointer.h>

/*****************************************************************************
 * about the object (what? the entity? no matter!)
 *****************************************************************************/

class Entity
{
public:
	Entity(int i) : _i(i)
	{}

	void set_i(int i)
	{
		_i = i;
	}

	int get_i() const
	{
		return _i;
	}

private:
	int _i;
};

typedef PVCore::PVSharedPtr<Entity> Entity_p;

extern Entity_p *shared_e;

#endif // MODAL_OBJ_H
