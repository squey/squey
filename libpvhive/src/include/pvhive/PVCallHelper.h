
#ifndef LIBPVHIVE_PVCALLHELPER_H
#define LIBPVHIVE_PVCALLHELPER_H

#include <pvkernel/core/PVFunctionTraits.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVActor.h>

namespace PVHive {

#define FUNC(Method) \
	decltype(&Method), &Method

#define FUNC_PROTOTYPE(RetType, Class, Method, Params...) \
	RetType(Class::*)(Params), &Class::Method

class PVCallHelper
{
public:
	template<typename F, F f, typename T, typename... P>
	static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(PVCore::PVSharedPtr<T>& object, P const& ... params)
	{
		PVActor<T> actor;
		PVHive::get().register_actor(object, actor);

		return actor.call<F, f>(params...);
	}

	template<typename F, F f, typename T, typename... P>
	static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(PVActor<T>& actor, P const& ... params)
	{
		return actor.call<F, f>(params...);
	}
};

template<typename F, F f, typename T, typename... P>
static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(T& obj, P const& ... params)
{
	return PVCallHelper::call<F, f>(obj, params...);
}

}

#endif /* LIBPVHIVE_PVCALLHELPER_H */
