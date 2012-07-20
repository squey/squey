
#ifndef LIVPVHIVE_PVACTOR_H
#define LIVPVHIVE_PVACTOR_H

#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

/**
 * @class PVActor
 *
 * A template class to specify actor on a given type/class.
 */
template <class T>
class PVActor : public PVActorBase
{
	friend class PVHive;

public:
	PVActor()
	{}

	/**
	 * Invoke an actor's method on its object
	 *
	 * @param params a variadic for method parameters
	 *
	 * @throw no_object
	 */
	template <typename F, F f, typename... P>
	void call(P const& ... params)
	{
		T *object = (T*)get_object();
		if (object != nullptr) {
			PVHive::get().call_object<T, F, f>(object, params...);
		} else {
			throw no_object("using an actor on a nullptr object");
		}
	}
};

// Helper macro to hide the C++ decltype verbosity
#define PVACTOR_CALL(Actor, Method, Param...)	  \
	(Actor).call<decltype(Method), Method>(Param)

}

#endif // LIVPVHIVE_PVACTOR_H
