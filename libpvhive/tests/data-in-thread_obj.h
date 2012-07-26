/**
 * \file data-in-thread_obj.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef entity_h
#define entity_h

#include <pvkernel/core/PVSharedPointer.h>

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

typedef PVCore::PVSharedptr<Entity> Entity_p;

extern Entity_p *static_e;

#endif // entity_h
