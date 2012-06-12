
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

	template <typename F, F f, typename... Ttypes>
	void call(Ttypes... params)
	{
		PVHive::get().call_object<T, F, f>(_object, params...);
	}

private:
	T *_object;
};

}

#endif // LIVPVHIVE_PVACTOR_H
