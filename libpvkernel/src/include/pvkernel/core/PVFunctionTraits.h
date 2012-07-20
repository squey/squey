#ifndef PVCORE_PVFUNCTIONTRAITS_H
#define PVCORE_PVFUNCTIONTRAITS_H

#include <pvkernel/core/PVTypeTraits.h>

namespace PVCore {

namespace PVTypeTraits {

// Function argument list traits
namespace __impl {

template <size_t N, typename T, typename... Tparams>
struct function_args_list_helper
{
	typedef function_args_list_helper<N+1, Tparams...> next_arg_type;
	typedef T arg_type;
	typedef typename add_pointer_const<arg_type>::type arg_pointer_const_type;
	typedef typename add_reference_const<arg_type>::type arg_reference_const_type;
	constexpr static size_t arg_n = N;

	template <size_t N_, typename T_, typename... Tparams_> friend class function_args_list_helper;

protected:
	typedef function_args_list_helper<N, T, Tparams...> this_type;

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
			return *p->_arg;
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
		_arg = &arg;
		_next_arg.set_args(others...);
	}


private:
	arg_pointer_const_type _arg;
	next_arg_type _next_arg;
};

template <size_t N, typename T>
struct function_args_list_helper<N, T>
{
	typedef T arg_type;
	typedef typename add_pointer_const<arg_type>::type arg_pointer_const_type;
	typedef typename add_reference_const<arg_type>::type arg_reference_const_type;
	constexpr static size_t arg_n = N;

	template <size_t N_, typename T_, typename... Tparams_> friend class function_args_list_helper;

protected:
	typedef function_args_list_helper<N, T> this_type;

	template <size_t aI, bool zero = (aI == 0)>
	struct get_arg_impl;

	template <size_t aI>
	struct get_arg_impl<aI, true>
	{
		typedef arg_reference_const_type result_type;
		inline static result_type get_arg(this_type const* p)
		{
			return *p->_arg;
		}
	};

public:
	template <size_t I>
	inline typename get_arg_impl<I>::result_type get_arg() const { return get_arg_impl<I>::get_arg(this); }

	inline void set_args(arg_reference_const_type arg) { _arg = &arg; }

private:
	arg_pointer_const_type _arg;
};

struct function_no_args_helper
{
	// For API compatibility
	void set_args() { }
};

} // __impl

template <typename... Tparams>
struct function_args_list: public __impl::function_args_list_helper<0, Tparams...>
{
};

// Function traits

namespace __impl {

template<typename F>
struct function_traits_helper;

template<typename R, typename... Tparams>
struct function_traits_helper<R (*)(Tparams...)>
{
	typedef R result_type;
	typedef function_args_list<Tparams...> arguments_type;
	constexpr static size_t arity = variadic_param_count<Tparams...>::count; 

	template <size_t I>
	struct type_of_arg: public variadic_n<I, Tparams...>
	{ };
};

template<typename R>
struct function_traits_helper<R (*)()>
{
	typedef R result_type;
	typedef function_no_args_helper arguments_type;
	constexpr static size_t arity = 0;
};

template<typename R, typename... Tparams>
struct function_traits_helper<R (Tparams...)>: public function_traits_helper<R (*)(Tparams...)>
{ };

template<typename T, typename R, typename... Tparams>
struct function_traits_helper<R (T::*)(Tparams...)>: public function_traits_helper<R (*)(Tparams...)>
{
	typedef T class_type;
};

} // __impl

template <typename F>
struct function_traits: public __impl::function_traits_helper<F>
{ };

} // PVTypeTraits

} // PVCore

#endif
