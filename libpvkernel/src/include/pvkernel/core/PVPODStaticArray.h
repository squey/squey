/**
 * \file PVPODStaticArray.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef __PVCORE_PVPODSTATICARRAY_H__
#define __PVCORE_PVPODSTATICARRAY_H__


namespace PVCore {

template<class T, std::size_t N, T V>
class PVPODStaticArray {
public:
	T elems[N];    // fixed-size array of elements of type T

public:
	// type definitions
	typedef T              value_type;
	typedef T*             iterator;
	typedef const T*       const_iterator;
	typedef T&             reference;
	typedef const T&       const_reference;
	typedef std::size_t    size_type;
	typedef std::ptrdiff_t difference_type;

	PVPODStaticArray()
	{
		// Warning: V must fits on a byte !
		memset(&elems[0], V, sizeof(T)*N);
	}

	// operator[]
	reference operator[](size_type i)
	{
		assert( i < N);
		return elems[i];
	}

	const_reference operator[](size_type i) const
	{
		assert( i < N);
		return elems[i];
	}

	// at() with range check
	reference at(size_type i) { return elems[i]; }
	const_reference at(size_type i) const { return elems[i]; }

	// size is constant
	static size_type size() { return N; }
	static bool empty() { return false; }
	static size_type max_size() { return N; }
	enum { static_size = N };
};

}

#endif // __PVCORE_PVPODSTATICARRAY_H__
