#include <pvhive/PVObserver.h>
#include <pvhive/PVHive.h>

PVHive::PVObserverBase::~PVObserverBase()
{
	if (_object != nullptr) {
		PVHive::get().unregister_observer(*this);
	}
}
