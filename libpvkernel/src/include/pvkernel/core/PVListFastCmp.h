/**
 * \file PVListFastCmp.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVLISTFASTCMP_H
#define PVCORE_PVLISTFASTCMP_H

#include <pvkernel/core/stdint.h>
#include <boost/functional/hash.hpp>

namespace PVCore {

// No "LibKernelDecl", as this is a template class
template <typename V, size_t MAX_DEPTH>
class PVListFastCmp
{
public:
	PVListFastCmp():
		_cur_size(0),
		_hash(0)
	{ }

public:
	inline void clear()
	{
		_hash = 0;
		_cur_size = 0;
	}

	void push(V v)
	{
		if (_cur_size > MAX_DEPTH) {
			boost::hash_combine(_hash, v);
			return;
		}

		if (_cur_size == MAX_DEPTH) {
			for (size_t i = 0; i < MAX_DEPTH; i++) {
				boost::hash_combine(_hash, _v[i]);
			}
			boost::hash_combine(_hash, v);
			_cur_size++;
			return;
		}

		_v[_cur_size] = v;
		_cur_size++;
	}

	inline bool operator==(const PVListFastCmp& other) const
	{
		if (_cur_size >= MAX_DEPTH) {
			return _hash == other._hash;
		}

		for (size_t i = 0; i < _cur_size; i++) {
			if (_v[i] != other._v[i]) {
				return false;
			}
		}

		return true;
	}
private:
	V _v[MAX_DEPTH];
	size_t _cur_size;
	size_t _hash;
};

}

#endif
