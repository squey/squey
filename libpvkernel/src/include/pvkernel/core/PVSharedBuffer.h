/**
 * \file PVSharedBuffer.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVCORE_PVSHAREDBUFFER_H
#define PVCORE_PVSHAREDBUFFER_H

#include <pvkernel/core/PVAllocators.h>
#include <pvkernel/core/PVSharedPointer.h>

namespace PVCore
{

template <typename T, class Alloc = std::allocator<T> >
class PVSharedBuffer
{
	typedef PVCore::PVSharedPtr<T>                   data_ptr_t;
	typedef Alloc                                    allocator_type;

public:
	typedef T                                        value_type;
	typedef typename allocator_type::pointer         pointer;
	typedef typename allocator_type::const_pointer   const_pointer;
        typedef typename allocator_type::reference       reference;
        typedef typename allocator_type::const_reference const_reference;
        typedef typename allocator_type::size_type       size_type;
	typedef pointer                                  iterator;
        typedef const_pointer                            const_iterator;

public:
	PVSharedBuffer()
	{
		clear();
	}

	~PVSharedBuffer()
	{
		clear();
	}

	inline void clear()
	{
		_data = data_ptr_t();
		_index = 0;
	}

	inline void reserve(size_t n)
	{
		_data = data_ptr_t(allocator_type().allocate(n));
		_index = 0;
	}

	inline void shrink_to_fit()
	{
		pointer ptr = nullptr;
		if (_index) {
			ptr = allocator_type().allocate(_index);
			memcpy(ptr, _data.get(), _index * sizeof(value_type));
		}
		_data = data_ptr_t(ptr);
	}

public:
	inline pointer get()
	{
		return _data.get();
	}

	inline const_pointer get() const
	{
		return _data.get();
	}

	inline size_type size() const
	{
		return _index;
	}

	inline void set_size(const size_type s)
	{
		_index = s;
	}

public:
	inline const_reference at(const size_type i) const
	{
		return _data.get()[i];
	}

	inline reference at(const size_type i)
	{
		return _data.get()[i];
	}

	inline void push_back(const T &v)
	{
		_data.get()[_index] = v;
		++_index;
	}

public:
	inline PVSharedBuffer &operator=(const PVSharedBuffer &buffer)
	{
		_data = buffer._data;
		_index = buffer._index;
		return *this;
	}

public:
	iterator begin() { return get(); }
        const_iterator begin() const { return get(); }

	iterator end() { return get() + size(); }
        const_iterator end() const { return get() + size(); }

private:
	data_ptr_t _data;
	size_type  _index;

};

}

#endif // PVCORE_PVSHAREDBUFFER_H
