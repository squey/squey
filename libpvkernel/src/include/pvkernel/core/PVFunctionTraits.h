/**
 * \file PVFunctionTraits.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVFUNCTIONTRAITS_H
#define PVCORE_PVFUNCTIONTRAITS_H

#include <pvkernel/core/PVTypeTraits.h>
#include <type_traits>

namespace PVCore {

namespace PVTypeTraits {

template<int ...>
struct seq_n { };

template<int N, int ...S>
struct gen_seq_n: gen_seq_n<N-1, N-1, S...> { };

template<int ...S>
struct gen_seq_n<0, S...>
{
	typedef seq_n<S...> type;
};

// Function argument list traits
namespace __impl {

template <typename T>
struct argument_storage_pointer
{
	typedef T arg_type;

public:
	typedef typename add_pointer_const<arg_type>::type type;
	typedef typename add_reference_const<arg_type>::type ref_type;

public:
	inline static ref_type to_ref(type t) { return *t; }
	inline static void set_storage(type& _storage, ref_type arg)
	{
		_storage = &arg;
	}
};

template <typename T>
struct argument_storage_copy
{
	typedef typename std::remove_reference<T>::type arg_type;

public:
	typedef arg_type type;
	typedef typename add_reference_const<arg_type>::type ref_type;

public:
	inline static ref_type to_ref(type const& v) { return v; }
	inline static void set_storage(type& storage, ref_type arg)
	{
		storage = arg;
	}
};

template <typename T>
struct argument_storage_copy<T&>: public argument_storage_pointer<T&>
{ };

template <size_t N, template <class Y> class storage_traits, typename T, typename... Tparams>
struct function_args_list_helper
{
	typedef function_args_list_helper<N+1, storage_traits, Tparams...> next_arg_type;
	typedef T arg_type;
	//typedef typename add_pointer_const<arg_type>::type arg_pointer_const_type;
	//typedef typename add_reference_const<arg_type>::type arg_reference_const_type;
	typedef storage_traits<arg_type> arg_storage_traits;
	typedef typename arg_storage_traits::type arg_storage_type;
	typedef typename arg_storage_traits::ref_type arg_reference_const_type;
	constexpr static size_t arg_n = N;

	template <size_t N_, template <class Y_> class storage_traits_, typename T_, typename... Tparams_> friend class function_args_list_helper;

protected:
	typedef function_args_list_helper<N, storage_traits, T, Tparams...> this_type;

	template <size_t aI, bool zero = (aI == 0)>
	struct get_arg_impl
	{
		typedef typename add_reference_const<typename variadic_n<aI, T, Tparams...>::type>::type result_type;

		inline static result_type get_arg(this_type const* p)
		{
			typedef typename next_arg_type::template get_arg_impl<aI-1> get_next_arg_impl;
			return get_next_arg_impl::get_arg(&p->_next_arg);
		}
	};

	template <size_t aI>
	struct get_arg_impl<aI, true>
	{
		typedef arg_reference_const_type result_type;

		inline static arg_reference_const_type get_arg(this_type const* p)
		{
			return arg_storage_traits::to_ref(p->_arg);
		}
	};
	
public:
	template <size_t I>
	inline typename get_arg_impl<I>::result_type get_arg() const
	{
		return get_arg_impl<I>::get_arg(this);
	}

	inline void set_args(typename add_reference_const<arg_type>::type arg, Tparams const& ... others)
	{
		arg_storage_traits::set_storage(_arg, arg);
		_next_arg.set_args(others...);
	}


private:
	arg_storage_type _arg;
	next_arg_type _next_arg;
};

template <size_t N, template <class Y> class storage_traits, typename T>
struct function_args_list_helper<N, storage_traits, T>
{
	typedef T arg_type;
	typedef storage_traits<arg_type> arg_storage_traits;
	typedef typename storage_traits<arg_type>::type arg_storage_type;
	typedef typename storage_traits<arg_type>::ref_type arg_reference_const_type;
	constexpr static size_t arg_n = N;

	template <size_t N_, template <class Y_> class storage_traits_, typename T_, typename... Tparams_> friend class function_args_list_helper;

protected:
	typedef function_args_list_helper<N, storage_traits, T> this_type;

	template <size_t aI, bool zero = (aI == 0)>
	struct get_arg_impl;

	template <size_t aI>
	struct get_arg_impl<aI, true>
	{
		typedef arg_reference_const_type result_type;
		inline static result_type get_arg(this_type const* p)
		{
			return arg_storage_traits::to_ref(p->_arg);
		}
	};

public:
	template <size_t I>
	inline typename get_arg_impl<I>::result_type get_arg() const { return get_arg_impl<I>::get_arg(this); }

	inline void set_args(arg_reference_const_type arg) { arg_storage_traits::set_storage(_arg, arg); }

private:
	arg_storage_type _arg;
};

struct function_no_args_helper
{
	// For API compatibility
	void set_args() { }
};

} // __impl

template <template <class Y> class storage_traits, typename... Tparams>
struct function_args_list: public __impl::function_args_list_helper<0, storage_traits, Tparams...>
{
};

// Enable an function_args_list with copy-storage to be set from the same one with a pointer-storage
template <typename... Tparams>
class function_args_list<__impl::argument_storage_copy, Tparams...>: public __impl::function_args_list_helper<0, __impl::argument_storage_copy, Tparams...>
{
	typedef __impl::function_args_list_helper<0, __impl::argument_storage_copy, Tparams...> base_type;
	typedef __impl::function_args_list_helper<0, __impl::argument_storage_pointer, Tparams...> base_pointer_type;

public:
	/* AG: commented as we still want trivial constructors for this class !
	 *

	function_args_list():
		base_type()
	{ }

	function_args_list(base_pointer_type const& o):
		base_type()
	{
		copy_from(o);
	}

	function_args_list(function_args_list const& o):
		base_type(o)
	{
	}

	 *
	 */

public:
	function_args_list& operator=(base_pointer_type const& o) { copy_from(o); return *this; }
	function_args_list& operator=(function_args_list const& o)
	{
		if (&o != this) {
			base_type::operator=(o);
		}
		return *this;
	}

private:
	inline void copy_from(base_pointer_type const& o)
	{
		do_copy_from(o, typename gen_seq_n<sizeof...(Tparams)>::type());
	}

	template <int... S>
	inline void do_copy_from(base_pointer_type const& o, seq_n<S...>)
	{
		base_type::set_args(o.template get_arg<S>()...);
	}
};

// Function traits

namespace __impl {

template<typename F>
struct function_traits_helper;

template<typename R, typename... Tparams>
struct function_traits_helper<R (*)(Tparams...)>
{
	typedef R result_type;
	typedef function_args_list<argument_storage_pointer, Tparams...> arguments_type;
	//typedef std::tuple<Tparams...> arguments_type;
	constexpr static size_t arity = variadic_param_count<Tparams...>::count; 
	typedef R(*pointer_type)(Tparams...);
	typedef function_args_list<argument_storage_copy, Tparams...> arguments_deep_copy_type;
	//typedef std::tuple<Tparams...> arguments_deep_copy_type;

	template <size_t I>
	struct type_of_arg: public variadic_n<I, Tparams...>
	{ };
};

template<typename R>
struct function_traits_helper<R (*)()>
{
	typedef R result_type;
	typedef function_no_args_helper arguments_type;
	typedef function_no_args_helper arguments_deep_copy_type;
	constexpr static size_t arity = 0;
	typedef R(*pointer_type)();
};

template<typename R, typename... Tparams>
struct function_traits_helper<R (Tparams...)>: public function_traits_helper<R (*)(Tparams...)>
{ };

template<typename T, typename R, typename... Tparams>
struct function_traits_helper<R (T::*)(Tparams...)>: public function_traits_helper<R (*)(Tparams...)>
{
	typedef T class_type;
	typedef R (T::*pointer_type)(Tparams...);
	//typedef typename function_traits_helper<R (*)(Tparams...)>::arguments_type arguments_type;
	typedef typename function_traits_helper<R (*)(Tparams...)>::arguments_deep_copy_type arguments_deep_copy_type;
	typedef arguments_deep_copy_type arguments_type;
	constexpr static bool is_const = false;

	template <pointer_type f, template <class Y> class argument_storage, typename R_ = R, typename std::enable_if<std::is_same<R_, void>::value == false, int>::type = 0>
	inline static R call(T& obj, function_args_list<argument_storage, Tparams...> const& args)
	{
		return std::move(do_call<f>(obj, args, typename gen_seq_n<sizeof...(Tparams)>::type()));
	}

	template <pointer_type f, template <class Y> class argument_storage, typename R_ = R, typename std::enable_if<std::is_same<R_, void>::value == true, int>::type = 0>
	inline static void call(T& obj, function_args_list<argument_storage, Tparams...> const& args)
	{
		do_call<f>(obj, args, typename gen_seq_n<sizeof...(Tparams)>::type());
	}

private:
	template <pointer_type f, template <class Y> class argument_storage, int... S>
	inline static R do_call(T& obj, function_args_list<argument_storage, Tparams...> const& args, seq_n<S...>)
	{
		return (obj.*f)(args.template get_arg<S>()...);
	}
};

template<typename T, typename R, typename... Tparams>
struct function_traits_helper<R (T::*)(Tparams...) const>: public function_traits_helper<R (*)(Tparams...)>
{
	typedef T class_type;
	typedef R (T::*pointer_type)(Tparams...) const;
	//typedef typename function_traits_helper<R (*)(Tparams...)>::arguments_type arguments_type;
	typedef typename function_traits_helper<R (*)(Tparams...)>::arguments_deep_copy_type arguments_deep_copy_type;
	typedef arguments_deep_copy_type arguments_type;
	constexpr static bool is_const = true;
	
	template <pointer_type f, template <class Y> class argument_storage>
	inline static R call(T const& obj, function_args_list<argument_storage, Tparams...> const& args)
	{
		return std::move(do_call<f>(obj, args, typename gen_seq_n<sizeof...(Tparams)>::type()));
	}

private:
	template <pointer_type f, template <class Y> class argument_storage, int... S>
	inline static R do_call(T const& obj, function_args_list<argument_storage, Tparams...> const& args, seq_n<S...>)
	{
		return (obj.*f)(args.template get_arg<S>()...);
	}
};


} // __impl

template <typename F>
struct function_traits: public __impl::function_traits_helper<F>
{ };

} // PVTypeTraits

} // PVCore

namespace std
{
	template <size_t N, template <class Y> class storage_traits, typename... Tparams>
	inline auto get(PVCore::PVTypeTraits::function_args_list<storage_traits, Tparams...> const& args) -> decltype(args.template get_arg<N>())
	{
		return args.template get_arg<N>();
	}
} // std

#endif
