//! \file PVVector.h
//! $Id: $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVVECTOR_H
#define PICVIZ_PVVECTOR_H

#include <algorithm>

namespace Picviz {

template <class C, int INCREMENT = 1000>
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

	void reset()
	{
		_index = 0;
	}

	void clear()
	{
		if(_array) {
			free(_array);
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
			std::copy(v._array, v._array + _index, _array);
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
			return std::equal(_array, _array + _index, v._array);
		}
	}

private:
	void reallocate(const unsigned size)
	{
		_array = (C*) realloc(_array, (size) * sizeof(C));
		_size = size;
	}

private:
	C        *_array;
	unsigned  _size;
	unsigned  _index;
};

#endif // PICVIZ_PVVECTOR_H

