#include <pvhive/PVObserver.h>
#include <pvhive/PVHive.h>

PVHive::PVObserverBase::~PVObserverBase()
{
	PVHive::get().unregister_observer(*this);
}
