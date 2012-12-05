/**
 * \file massive_common.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef MASSIVE_COMMON_H
#define MASSIVE_COMMON_H

#include <pvhive/PVHive.h>
#include <pvhive/PVActor.h>
#include <pvhive/PVObserver.h>

#include <tbb/tick_count.h>

/*****************************************************************************
 * datas
 *****************************************************************************/

class Property
{
public:
	Property(int v = 0) : _value(v)
	{}

	int get_value() const
	{
		return _value;
	}

	void set_value(int v)
	{
		_value = v;
	}

private:
	int _value;
};

class Block
{
public:
	Block(int n) : _count(n)
	{
		_array = new Property [n];
	}

	~Block()
	{
		if (_array) {
			delete [] _array;
			_array = nullptr;
		}
	}

	void set_value(int v)
	{
		_value = v;
	}

	int get_value() const
	{
		return _value;
	}

	void set_prop(int i, Property p)
	{
		_array[i] = p;
	}

	const Property& get_prop(int i) const
	{
		return _array[i];
	}

	Property& get_prop(int i)
	{
		return _array[i];
	}

private:
	Property *_array;
	int       _count;
	int       _value;
};

typedef PVCore::PVSharedPtr<Block> Block_p;


/*****************************************************************************
 * actors
 *****************************************************************************/

class BlockAct : public PVHive::PVActor<Block>
{
public:
	BlockAct() : _value(0)
	{}

	BlockAct(int value) : _value(value)
	{}

	int get_value() const
	{
		return _value;
	}

	void action();
private:
	int _value;
};

class PropertyAct : public PVHive::PVActor<Block>
{
public:
	PropertyAct() : _index(0), _value(0)
	{}

	PropertyAct(int i, int v) : _index(i), _value(v)
	{}

	int get_value() const
	{
		return _value;
	}

	int get_index() const
	{
		return _index;
	}

	void action();
private:
	int _index;
	int _value;
};

/*****************************************************************************
 * observers
 *****************************************************************************/

class BlockObs : public PVHive::PVObserver<Block>
{
public:
	BlockObs()
	{}

	int get_value() const
	{
		return _value;
	}

	void refresh()
	{
		_value = get_object()->get_value();
	}

	void about_to_be_deleted()
	{
		_quit = true;
	}

private:
	int _value;
	bool _quit;
};

class PropertyObs : public PVHive::PVObserver<Property>
{
public:
	PropertyObs() : _quit(false)
	{}

	int get_value() const
	{
		return _value;
	}

	void refresh()
	{
		_value = get_object()->get_value();
	}

	void about_to_be_deleted()
	{
		_quit = true;
	}

private:
	int _value;
	bool _quit;
};

/*****************************************************************************
 * property accessor
 *****************************************************************************/

Property* get_prop(Block& b, int i);

#endif // MASSIVE_COMMON_H
