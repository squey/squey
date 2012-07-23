/**
 * \file PVTypeTraits.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVTYPETRAITS_H
#define PVCORE_PVTYPETRAITS_H

#include <boost/shared_ptr.hpp>
#include <boost/type_traits.hpp>

namespace PVCore {

namespace PVTypeTraits {

template <class T>
struct is_shared_ptr: public boost::false_type
{ };

template <class T>
struct is_shared_ptr<boost::shared_ptr<T> >: public boost::true_type
{ };

template <class T>
struct add_shared_ptr
{
	typedef boost::shared_ptr<T> type;
};

template <class T>
struct add_shared_ptr<boost::shared_ptr<T> >
{
	typedef T type;
};

template <class T>
struct remove_shared_ptr
{
	typedef T type;
};

template <class T>
struct remove_shared_ptr<boost::shared_ptr<T> >
{
	typedef T type;
};

// Dynamic cast T* into Y*
template <class Y, class T>
struct dynamic_pointer_cast
{
	static Y cast(T p) { return dynamic_cast<T>(p); }
};

template <class Y, class T>
struct dynamic_pointer_cast<boost::shared_ptr<Y>, boost::shared_ptr<T> >
{
	typedef typename boost::shared_ptr<T> org_pointer;
	typedef typename boost::shared_ptr<Y> result_pointer;
	static result_pointer cast(org_pointer const& p) { return boost::dynamic_pointer_cast<Y>(p); }
};

// Get a pointer from whatever type
template <class T>
struct pointer
{
	typedef typename boost::remove_reference<T>::type type_noref;
	typedef type_noref* type;
	static inline type get(type_noref& obj)
	{
		return &obj;
	}
};

template <class T>
struct pointer<T*>
{
	typedef T* type;
	static inline type get(T* obj) { return obj; }
};

template <class T>
struct pointer<boost::shared_ptr<T> >
{
	typedef boost::shared_ptr<T> type;
	static inline type get(type obj) { return obj; }
};

template <class T>
struct pointer<boost::shared_ptr<T>& >
{
	typedef boost::shared_ptr<T>& type;
	static inline type get(type obj) { return obj; }
};

namespace __impl
{
	template <std::size_t mod>
	struct is_size_multiple_impl
	{
		static const bool result = false;
	};

	template <>
	struct is_size_multiple_impl<0u>
	{
		static const bool result = true;
	};
}

template <class T, class MultipleOf>
struct is_size_multiple
{
	static const bool value = __impl::is_size_multiple_impl<sizeof(T)%sizeof(MultipleOf)>::result;
};

template <class A, class B>
struct bigger_than
{
	static const bool value = sizeof(A)>sizeof(B);
};

}

}

#endif
