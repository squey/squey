
#ifndef LIVPVHIVE_PVACTOR_H
#define LIVPVHIVE_PVACTOR_H

#include <pvhive/PVHive.h>

namespace PVHive
{

class PVActorBase {};

template <class T>
class PVActor : public PVActorBase
{
public:
	friend class PVHive;

	~PVActor()
	{
		PVHive::get().unregister_actor(*this);
	}

	/**
	 * Invoke an actor's method on its object
	 *
	 * @param params the method parameters
	 */
	template <typename F, F f, typename... P>
	void call(P... params)
	{
		PVHive::get().call_object<T, F, f>(_object, params...);
	}

private:
	T *_object;
};

}

#endif // LIVPVHIVE_PVACTOR_H
