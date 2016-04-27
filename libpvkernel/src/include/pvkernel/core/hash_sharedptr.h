/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef HASH_SHAREDPTR_H
#define HASH_SHAREDPTR_H

#ifdef QHASH_H
#error libpvkernel/core/hash_sharedptr.h must be included before QHash if you want to use it as a QHash key.
#endif

#include <memory>
#include <cstdint>

// Taken from Qt's qhash.h
template <class T> inline unsigned int qHash(std::shared_ptr<T> const& p)
{
	uintptr_t key = (uintptr_t)p.get();
	if (sizeof(uintptr_t) > sizeof(unsigned int)) {
		return (unsigned int)(((key >> (8 * sizeof(unsigned int) - 1)) ^ key) & (~0U));
	} else {
		return (unsigned int)(key & (~0U));
	}
}

#endif
