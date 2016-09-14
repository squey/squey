/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVTYPETRAITS_H
#define PVCORE_PVTYPETRAITS_H

#include <memory>
#include <type_traits>

namespace PVCore
{

namespace PVTypeTraits
{

template <class T>
struct remove_shared_ptr {
	typedef T type;
};

template <class T>
struct remove_shared_ptr<std::shared_ptr<T>> {
	typedef T type;
};

// Get a pointer from whatever type
template <class T>
struct pointer {
	typedef typename std::remove_reference<T>::type type_noref;
	typedef type_noref* type;
	static inline type get(type_noref& obj) { return &obj; }
};

template <class T>
struct pointer<T*> {
	typedef T* type;
	static inline type get(T* obj) { return obj; }
};

template <class T>
struct pointer<std::shared_ptr<T>> {
	typedef std::shared_ptr<T> type;
	static inline type get(type obj) { return obj; }
};

template <class T>
struct pointer<std::shared_ptr<T>&> {
	typedef std::shared_ptr<T>& type;
	static inline type get(type obj) { return obj; }
};
} // namespace PVTypeTraits
} // namespace PVCore

#endif
