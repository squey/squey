#ifndef PVHIVE_PVWAX_H
#define PVHIVE_PVWAX_H

#include <pvkernel/core/PVFunctionTraits.h>

/**
 * @attention any specialization of template method ::call_object() *must* be
 * declared in the namespace PVHive (you can use the macros
 * PVHIVE_CALL_OBJECT_BLOCK_BEGIN() and PVHIVE_CALL_OBJECT_BLOCK_END()).
 */

#define PVHIVE_CALL_OBJECT_BLOCK_BEGIN() namespace PVHive {
#define PVHIVE_CALL_OBJECT_BLOCK_END() }

#define IMPL_WAX(Method, obj, args) \
	template <> \
	typename PVCore::PVTypeTraits::function_traits<decltype(&Method)>::result_type PVHive::call_object<decltype(&Method), &Method>(typename PVCore::PVTypeTraits::function_traits<decltype(&Method)>::class_type* obj, typename PVCore::PVTypeTraits::function_traits<decltype(&Method)>::arguments_type const& args)

/*
#define IMPL_WAX_OVERLOAD(RetType, Class, Method, obj, args, Params...) \
	template <> \
	RetType PVHive::call_object<RetType(Class::*)(Params), &Class::Method>(Class* obj, typename PVCore::PVTypeTraits::function_traits<RetType(Class::*)(Params)>::arguments_type const& args)
*/

#define DECLARE_WAX(Method) \
	PVHIVE_CALL_OBJECT_BLOCK_BEGIN()\
	IMPL_WAX(Method, obj, args);\
	PVHIVE_CALL_OBJECT_BLOCK_END()
/*
#define DECLARE_WAX_OVERLOAD(RetType, Class, Method, Params...) \
	PVHIVE_CALL_OBJECT_BLOCK_BEGIN()\
	IMPL_WAX_OVERLOAD(RetType, Class, Method, obj, args, Params...);\
	PVHIVE_CALL_OBJECT_BLOCK_END()
*/

#endif
