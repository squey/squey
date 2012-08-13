/**
 * \file PVVector.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVVECTOR_H
#define PICVIZ_PVVECTOR_H

#include <pvkernel/core/PVAllocators.h>

// for memcpy()
#include <string.h>

namespace Picviz {

template <class C, int INCREMENT = 1000, class Alloc = PVCore::PVReallocableCAllocator<C> >
class PVVector
{
public:
	PVVector() :
		_array(0),
		_size(0),
		_index(0)
	{}

	PVVector(const unsigned size) :
		_array(0),
		_index(0)
	{
		if(size != 0) {
			reallocate(size);
		} else {
			_size = size;
		}
	}

	~PVVector()
	{
		clear();
	}

	void reserve(const unsigned size)
	{
		reallocate(size);
	}

	void compact()
	{
		reallocate(_index);
	}

	void reset()
	{
		_index = 0;
	}

	void clear()
	{
		if(_array) {
			Alloc().deallocate(_array, _size);
			_array = 0;
			_size = 0;
			_index = 0;
		}
	}

	inline unsigned size() const
	{
		return _index;
	}

	inline unsigned capacity() const
	{
		return _size;
	}

	inline size_t memory() const
	{
		return sizeof(PVVector) + _size * sizeof(C);
	}

	inline bool is_null() const
	{
		return (_array == 0);
	}

	inline C &at(const int i)
	{
		return _array[i];
	}

	inline C const& at(const int i) const
	{
		return _array[i];
	}

	inline void push_back(const C &c)
	{
		if (_index == _size) {
			reallocate(_size + INCREMENT);
		}
		_array[_index++] = c;
	}

	PVVector<C> &operator=(const PVVector<C> &v)
	{
		clear();
		if(v._size) {
			_index = v._index;
			reallocate(v._size);
			memcpy(_array, v._array, _index * sizeof(C));
		}
		return *this;
	}

	bool operator==(const PVVector<C> &v) const
	{
		if(_index != v._index) {
			return false;
		} else if(_array == 0) {
			return (v._array == 0);
		} else if(v._array == 0) {
			return false;
		} else {
			return (memcmp(_array, v._array, _index * sizeof(C)) == 0);
		}
	}

private:
	void reallocate(const unsigned size)
	{
		_array = Alloc().reallocate(_array, _size, size);
		_size = size;
	}

private:
	C        *_array;
	unsigned  _size;
	unsigned  _index;
};

}

#endif // PICVIZ_PVVECTOR_H

