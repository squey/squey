#ifndef PVHIVE_PVWAX_H
#define PVHIVE_PVWAX_H

#include <pvkernel/core/PVFunctionTraits.h>
#include <pvhive/PVCallHelper.h>

/**
 * @attention any specialization of template method ::call_object() *must* be
 * declared in the namespace PVHive (you can use the macros
 * PVHIVE_CALL_OBJECT_BLOCK_BEGIN() and PVHIVE_CALL_OBJECT_BLOCK_END()).
 */

#define PVHIVE_CALL_OBJECT_BLOCK_BEGIN() namespace PVHive {
#define PVHIVE_CALL_OBJECT_BLOCK_END() }

#define IMPL_WAX(Method, obj, args) \
	template <> \
	void PVHive::call_object<decltype(&Method), &Method>(typename PVCore::PVTypeTraits::function_traits<decltype(&Method)>::class_type* obj, typename PVCore::PVTypeTraits::function_traits<decltype(&Method)>::arguments_type const& args)

#define DECLARE_WAX(Method) \
	PVHIVE_CALL_OBJECT_BLOCK_BEGIN()\
	IMPL_WAX(Method, obj, args);\
	PVHIVE_CALL_OBJECT_BLOCK_END()

#endif
