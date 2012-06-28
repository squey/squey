
#include <QMetaType>

#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVObserver.h>

/*****************************************************************************/

PVHive::PVHive *PVHive::PVHive::_hive = nullptr;

/*****************************************************************************
 * PVHive::PVHive::unregister_observer()
 *****************************************************************************/

void PVHive::PVHive::unregister_observer(PVObserverBase& observer)
{
	// the observer must have a valid object
	assert(observer._object != nullptr);

	write_lock_t write_lock(_observables_lock);
	auto entry = _observables.find(observer._object);
	if (entry != _observables.end()) {
		entry->second.observers.erase(&observer);
	}
	observer._object = nullptr;
}

/*****************************************************************************
 * PVHive::PVHive::unregister_object()
 *****************************************************************************/

void PVHive::PVHive::unregister_object(void *object)
{
	// the object must be a valid address
	assert(object != nullptr);

	read_lock_t read_lock(_observables_lock);
	auto entry = _observables.find(object);
	if (entry != _observables.end()) {
		// notify properties observers
		for (auto it : entry->second.properties) {
			auto res = _observables.find(it);
			for (auto pit : res->second.observers) {
				pit->about_to_be_deleted();
			}
			_observables.erase(it);
		}
		// notify observers
		for (auto it : entry->second.observers) {
			it->about_to_be_deleted();
		}
		_observables.erase(object);
	}
}

/*****************************************************************************
 * PVHive::PVHive::do_refresh_observers()
 *****************************************************************************/
void PVHive::PVHive::do_refresh_observers(void *object)
{
	read_lock_t read_lock(_observables_lock);
	auto entry = _observables.find(object);
	if (entry != _observables.end()) {
		for (auto it : entry->second.observers) {
			it->refresh();
		}
	}
}
