
#include <QMetaType>

#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVObserver.h>

/*****************************************************************************/

PVHive::PVHive *PVHive::PVHive::_hive = nullptr;

/*****************************************************************************
 * PVHive::PVHive::unregister_actor()
 *****************************************************************************/

void PVHive::PVHive::unregister_actor(PVActorBase& actor)
{
	// the actor must have a valid object
	assert(actor._object != nullptr);

	{
		write_lock_t write_lock(_observables_lock);
		auto entry = _observables.find(actor._object);
		if (entry != _observables.end()) {
			entry->second.actors.erase(&actor);
		}
	}

	actor.set_object(nullptr);
}

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

	// first, actors have to be detached from their objects
	{
		read_lock_t read_lock(_observables_lock);
		auto entry = _observables.find(object);
		if (entry != _observables.end()) {
			for (auto it : entry->second.properties) {
				auto res = _observables.find(it);
				if (res != _observables.end()) {
					for (auto pit : res->second.actors) {
						pit->set_object(nullptr);
					}
				}
			}
			for (auto it : entry->second.actors) {
				it->set_object(nullptr);
			}
		}
	}

	// finally, we release data related to object
	write_lock_t write_lock(_observables_lock);
	auto entry = _observables.find(object);
	if (entry != _observables.end()) {
		// notify properties observers
		for (auto it : entry->second.properties) {
			auto res = _observables.find(it);
			if (res != _observables.end()) {
				for (auto pit : res->second.observers) {
					pit->about_to_be_deleted();
				}
				_observables.erase(it);
			}
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
