
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

	void set_prop1(Property p)
	{
		_array[0] = p;
	}

	const Property& get_prop(int i) const
	{
		if (i < 0) {
			throw std::out_of_range("bad range for property access");
		} else if (i >= _count) {
			throw std::out_of_range("bad range for property access");
		} else {
			return _array[i];
		}
	}

private:
	Property *_array;
	int       _count;
	int       _value;
};

typedef PVCore::pv_shared_ptr<Block> Block_p;


/*****************************************************************************
 * actors
 *****************************************************************************/

class BlockAct : public PVHive::PVActor<Block>
{
public:
	BlockAct()
	{}

	void action()
	{
		PVACTOR_CALL(*this, &Block::set_value, rand());
	}

};

class PropertyAct : public PVHive::PVActor<Block>
{
public:
	PropertyAct() : _index(0)
	{}

	PropertyAct(int i) : _index(i)
	{}

	void action()
	{
		PVACTOR_CALL(*this, &Block::set_prop, _index, Property(rand()));
	}
private:
	int _index;
};

/*****************************************************************************
 * observers
 *****************************************************************************/

class BlockObs : public PVHive::PVObserver<Block>
{
public:
	BlockObs()
	{}

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

const Property& get_prop(const Block& b, int i);


/*****************************************************************************
 * display stats
 *****************************************************************************/

void print_stat(const char *what, tbb::tick_count t1, tbb::tick_count t2, long num);

#endif // MASSIVE_COMMON_H
