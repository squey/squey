/**
 * \file hash_sharedptr.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef HASH_SHAREDPTR_H
#define HASH_SHAREDPTR_H

#ifdef QHASH_H
#error libpvkernel/core/hash_sharedptr.h must be included before QHash if you want to use it as a QHash key.
#endif

#include <pvkernel/core/stdint.h>
#include <boost/shared_ptr.hpp>

// Taken from Qt's qhash.h
template <class T>
inline unsigned int qHash(boost::shared_ptr<T> const& p)
{
	uintptr_t key = (uintptr_t)p.get();
	if (sizeof(uintptr_t) > sizeof(unsigned int)) {
		return (unsigned int)(((key >> (8 * sizeof(unsigned int) - 1)) ^ key) & (~0U));
	} else {
		return (unsigned int)(key & (~0U));
	}    

}

#endif
