
#ifndef LIVPVHIVE_PVACTOR_H
#define LIVPVHIVE_PVACTOR_H

#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>

#include <pvkernel/core/PVSpinLock.h>

namespace PVHive
{

template <class T>
class PVActor : public PVActorBase
{
public:
	friend class PVHive;

	PVActor()
	{}

	/**
	 * Invoke an actor's method on its object
	 *
	 * @param params the method parameters
	 */
	template <typename F, F f, typename... P>
	void call(P... params)
	{
		PVCore::pv_spin_lock_guard_t slg(_spinlock);
		PVHive::get().call_object<T, F, f>((T*)get_object(), params...);
	}
};

// a little macro to hide the decltype verbosity
#define PVACTOR_CALL(Actor, Method, Param...)	  \
	(Actor).call<decltype(Method), Method>(Param)

}

#endif // LIVPVHIVE_PVACTOR_H
