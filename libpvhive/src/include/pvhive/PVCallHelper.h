
#ifndef LIBPVHIVE_PVCALLHELPER_H
#define LIBPVHIVE_PVCALLHELPER_H

#include <pvkernel/core/PVFunctionTraits.h>
#include <pvkernel/core/PVSharedPointer.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVActor.h>

namespace PVHive {

class PVCallHelper
{
public:
	template<typename F, F f, typename T, typename... P>
	static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(PVCore::PVSharedPtr<T>& object, P && ... params)
	{
		PVActor<T> actor;
		PVHive::get().register_actor(object, actor);

		return actor.call<F, f>(std::forward<P>(params)...);
	}

	template<typename F, F f, typename T, typename... P>
	static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(PVActor<T>& actor, P && ... params)
	{
		return actor.call<F, f>(std::forward<P>(params)...);
	}
};

template<typename F, F f, typename T, typename... P>
static typename PVCore::PVTypeTraits::function_traits<F>::result_type call(T& obj, P && ... params)
{
	return PVCallHelper::call<F, f>(obj, std::forward<P>(params)...);
}

template <typename F, F f>
struct PVHiveFuncCaller
{
	typedef F func_t;

	template <typename T, typename... P>
	static typename PVCore::PVTypeTraits::function_traits<func_t>::result_type call(PVCore::PVSharedPtr<T>& object, P && ... params)
	{
		return PVCallHelper::call<F, f>(object, std::forward<P>(params)...);
	}
};

}

#endif /* LIBPVHIVE_PVCALLHELPER_H */
