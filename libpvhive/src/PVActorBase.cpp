#include <pvhive/PVActorBase.h>
#include <pvhive/PVHive.h>

PVHive::PVActorBase::~PVActorBase()
{
	PVHive::get().unregister_actor(*this);
}
