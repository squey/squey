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

// Dynamic cast T* into Y*
template <class Y, class T>
struct dynamic_pointer_cast {
	static Y cast(T p) { return dynamic_cast<T>(p); }
};

template <class Y, class T>
struct dynamic_pointer_cast<std::shared_ptr<Y>, std::shared_ptr<T>> {
	typedef typename std::shared_ptr<T> org_pointer;
	typedef typename std::shared_ptr<Y> result_pointer;
	static result_pointer cast(org_pointer const& p) { return std::dynamic_pointer_cast<Y>(p); }
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

template <class T, class MultipleOf>
struct is_size_multiple {
	static const bool value = not(sizeof(T) % sizeof(MultipleOf));
};

template <class A, class B>
struct bigger_than {
	static const bool value = sizeof(A) > sizeof(B);
};

// Const forwarder
// Make a type const iif another type is const
template <class T, class Tref>
struct const_fwd {
	typedef T type;
};

template <class T, class Tref>
struct const_fwd<T, const Tref> {
	typedef typename std::add_const<T>::type type;
};

template <class T, class Tref>
struct const_fwd<T, const Tref&> {
	typedef typename std::add_const<T>::type type;
};

template <class T, class Tref>
struct const_fwd<T&, const Tref> {
	typedef typename std::add_const<T>::type& type;
};

template <class T, class Tref>
struct const_fwd<T&, const Tref&> {
	typedef typename std::add_const<T>::type& type;
};

// Polymorhpic object helpers
template <typename T, typename std::enable_if<std::is_polymorphic<T>::value == true, int>::type = 0>
inline typename const_fwd<void, T>::type* get_starting_address(T* obj)
{
	return dynamic_cast<typename const_fwd<void, T>::type*>(obj);
}

template <typename T,
          typename std::enable_if<std::is_polymorphic<T>::value == false, int>::type = 0>
inline typename const_fwd<void, T>::type* get_starting_address(T* obj)
{
	return reinterpret_cast<typename const_fwd<void, T>::type*>(obj);
}
}
}

#endif
