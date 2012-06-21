
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

	write_lock_t write_lock(_observers_lock);
	auto res = _observers.find(observer._object);
	if (res != _observers.end()) {
		res->second.erase(&observer);
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

	read_lock_t read_lock(_observers_lock);
	auto res = _observers.find(object);
	if (res != _observers.end()) {
		for (auto it : res->second) {
			it->about_to_be_deleted();
		}
		_observers.erase(object);
	}
}

/*****************************************************************************
 * PVHive::PVHive::do_refresh_observers()
 *****************************************************************************/
void PVHive::PVHive::do_refresh_observers(void *object)
{
	read_lock_t read_lock(_observers_lock);
	auto res = _observers.find(object);
	if (res != _observers.end()) {
		for (auto it : res->second) {
			it->refresh();
		}
	}
}
