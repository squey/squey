/**
 * \file PVTypeTraits.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVTYPETRAITS_H
#define PVCORE_PVTYPETRAITS_H

#include <memory>
#include <type_traits>

#include <boost/call_traits.hpp>

#include <iostream>

namespace PVCore {

namespace PVTypeTraits {

template <class T>
struct is_shared_ptr: public std::false_type
{ };

template <class T>
struct is_shared_ptr<std::shared_ptr<T> >: public std::true_type
{ };

template <class T>
struct add_shared_ptr
{
	typedef std::shared_ptr<T> type;
};

template <class T>
struct add_shared_ptr<std::shared_ptr<T> >
{
	typedef T type;
};

template <class T>
struct remove_shared_ptr
{
	typedef T type;
};

template <class T>
struct remove_shared_ptr<std::shared_ptr<T> >
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
struct dynamic_pointer_cast<std::shared_ptr<Y>, std::shared_ptr<T> >
{
	typedef typename std::shared_ptr<T> org_pointer;
	typedef typename std::shared_ptr<Y> result_pointer;
	static result_pointer cast(org_pointer const& p) { return std::dynamic_pointer_cast<Y>(p); }
};

// Get a pointer from whatever type
template <class T>
struct pointer
{
	typedef typename std::remove_reference<T>::type type_noref;
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
struct pointer<std::shared_ptr<T> >
{
	typedef std::shared_ptr<T> type;
	static inline type get(type obj) { return obj; }
};

template <class T>
struct pointer<std::shared_ptr<T>& >
{
	typedef std::shared_ptr<T>& type;
	static inline type get(type obj) { return obj; }
};

template <class T>
inline T* get_pointer(T& p) { return &p; }

template <class T>
inline T const* get_pointer(T const& p) { return &p; }

template <class T>
inline T* get_pointer(T* p) { return p; }

template <class T>
inline T const* get_pointer(T const* p) { return p; }

template <class T>
inline T* get_pointer(std::shared_ptr<T> const& p) { return p.get(); }

template <class T>
inline T const* get_pointer(std::shared_ptr<T const> const& p) { return p.get(); }

namespace __impl {
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

template <typename T>
struct add_pointer_const
{
	typedef typename std::add_pointer<typename std::add_const<T>::type>::type type;
};

template <typename T>
struct add_reference_const
{
	typedef typename std::add_lvalue_reference<typename std::add_const<T>::type>::type type;
};

// Const forwarder
// Make a type const iif another type is const
template <class T, class Tref>
struct const_fwd
{
	typedef T type;
};

template <class T, class Tref>
struct const_fwd<T, const Tref>
{
	typedef typename std::add_const<T>::type type;
};

template <class T, class Tref>
struct const_fwd<T, const Tref&>
{
	typedef typename std::add_const<T>::type type;
};

template <class T, class Tref>
struct const_fwd<T&, const Tref>
{
	typedef typename std::add_const<T>::type& type;
};

template <class T, class Tref>
struct const_fwd<T&, const Tref&>
{
	typedef typename std::add_const<T>::type& type;
};

// Polymorhpic object helpers
template <typename T, typename std::enable_if<std::is_polymorphic<T>::value == true, int>::type = 0>
inline typename const_fwd<void, T>::type* get_starting_address(T* obj)
{
	return dynamic_cast<typename const_fwd<void, T>::type*>(obj);
}

template <typename T, typename std::enable_if<std::is_polymorphic<T>::value == false, int>::type = 0>
inline typename const_fwd<void, T>::type* get_starting_address(T* obj)
{
	return reinterpret_cast<typename const_fwd<void, T>::type*>(obj);
}

template <typename U, typename T, typename std::enable_if<std::is_polymorphic<T>::value == true, int>::type = 0>
inline typename const_fwd<typename std::remove_pointer<U>::type, T>::type* dynamic_cast_if_possible(T* obj)
{
	return dynamic_cast<typename const_fwd<typename std::remove_pointer<U>::type, T>::type*>(obj);
}

template <typename U, typename T, typename std::enable_if<std::is_polymorphic<T>::value == false, int>::type = 0>
inline typename const_fwd<typename std::remove_pointer<U>::type, T>::type* dynamic_cast_if_possible(T*)
{
	return nullptr;
}

template <class T, class Tref>
typename const_fwd<T, typename std::remove_reference<typename boost::call_traits<Tref>::param_type>::type>::type&& forward_with_const(typename std::remove_reference<T>::type& t)
{
	typedef typename std::remove_reference<typename boost::call_traits<Tref>::param_type>::type ref_type;
	std::cout << std::is_const<ref_type>::value << std::endl;
	return std::forward<typename const_fwd<T, ref_type>::type>(t);
}

template <class T, class Tref>
typename const_fwd<T, typename std::remove_reference<typename boost::call_traits<Tref>::param_type>::type>::type&& forward_with_const(typename std::remove_reference<T>::type&& t)
{
	typedef typename std::remove_reference<typename boost::call_traits<Tref>::param_type>::type ref_type;
	std::cout << std::is_const<ref_type>::value << std::endl;
	return std::forward<typename const_fwd<T, ref_type>::type>(t);
}

}

// Variadic informations

namespace __impl {
template <size_t N, typename T, typename... Tparams>
struct variadic_param_count_helper
{
	constexpr static size_t count = variadic_param_count_helper<N+1, Tparams...>::count;
};

template <size_t N, typename T>
struct variadic_param_count_helper<N, T>
{
	constexpr static size_t count = N+1;
};

template <size_t I, size_t N, typename T, typename... Tparams>
struct variadic_n_helper
{
	typedef typename variadic_n_helper<I, N+1, Tparams...>::type type;
};

template <size_t I, typename T, typename... Tparams>
struct variadic_n_helper<I, I, T, Tparams...>
{
	typedef T type;
};
}

/*! \brief Get the number of elements of variadic template parameters
 */
template <typename... Tparams>
struct variadic_param_count
{
	constexpr static size_t count = __impl::variadic_param_count_helper<0, Tparams...>::count;
};

/*! \brief Get the I'th type of variadic template parameters
 */
template <size_t I, typename... Tparams>
struct variadic_n
{
	typedef typename __impl::variadic_n_helper<I, 0, Tparams...>::type type;
};


}

#endif
