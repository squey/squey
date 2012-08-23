/**
 * \file test_adrien_objs.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef TEST_ADRIEN_OBJS_H
#define TEST_ADRIEN_OBJS_H

#include <pvkernel/core/PVSharedPointer.h>

class ObjectProperty
{
public:
	ObjectProperty(int v = 0): _v(v)
	{ }
public:
	int _v;
};

class MyObject
{
public:
	MyObject(int i): _i(i) { }

public:
	int const& get_i() const { return _i; };
	void set_i(int i) { _i = i; }
	void set_i2(int const& i) { _i = i; }

	void set_prop(ObjectProperty const& p)
	{
		std::cout << "set_prop with ObjectProperty: " << &p << " with i=" << p._v << std::endl;
		_prop = p;
	}
	ObjectProperty const& get_prop() const { return _prop; }

private:
	int _i;
	ObjectProperty _prop;
};

typedef PVCore::PVSharedPtr<MyObject> MyObject_p;

#endif // TEST_ADRIEN_OBJS_H
