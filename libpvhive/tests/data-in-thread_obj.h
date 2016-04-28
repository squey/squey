/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef entity_h
#define entity_h

#include <pvkernel/core/PVSharedPointer.h>

class Entity
{
  public:
	Entity(int i) : _i(i) {}

	void set_i(int i) { _i = i; }

	int get_i() const { return _i; }

  private:
	int _i;
};

typedef PVCore::PVSharedPtr<Entity> Entity_p;

extern Entity_p* static_e;

#endif // entity_h
