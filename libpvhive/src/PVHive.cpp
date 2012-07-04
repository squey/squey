
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

	observables_t::accessor acc;

	if (_observables.find(acc, actor._object)) {
		acc->second.actors.erase(&actor);
		if (acc->second.empty()) {
			acc.release();
			_observables.erase(actor._object);
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

	observables_t::accessor acc;

	if (_observables.find(acc, observer._object)) {
		acc->second.observers.erase(&observer);
		if (acc->second.empty()) {
			acc.release();
			_observables.erase(observer._object);
		}
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

	observables_t::accessor acc;

	if (_observables.find(acc, object)) {

		// unregistering properties
		for (auto it : acc->second.properties) {
			observables_t::accessor pacc;
			if (_observables.find(pacc, it)) {
				// actors...
				for (auto pit : pacc->second.actors) {
					pit->set_object(nullptr);
				}

				// and observers
				for (auto pit : pacc->second.observers) {
					pit->about_to_be_deleted();
				}
			}
		}

		// unregistering actors...
		for (auto it : acc->second.actors) {
			it->set_object(nullptr);
		}

		// and observers
		for (auto it : acc->second.observers) {
			it->about_to_be_deleted();
		}

		// finally, the entry of object is removed
		acc.release();
		_observables.erase(object);
	}
}

/*****************************************************************************
 * PVHive::PVHive::do_refresh_observers()
 *****************************************************************************/
void PVHive::PVHive::do_refresh_observers(void *object)
{
	observables_t::const_accessor acc;

	if (_observables.find(acc, object)) {
		for (auto it : acc->second.observers) {
			it->refresh();
		}
	}
}
