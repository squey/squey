/**
 * \file PVHive.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QMetaType>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVDataTreeObject.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVObserver.h>

/*****************************************************************************/

PVHive::PVHive *PVHive::PVHive::_hive = nullptr;

/*****************************************************************************
 * PVHive::PVHive::unregister_actor()
 *****************************************************************************/

bool PVHive::PVHive::unregister_actor(PVActorBase& actor)
{
	if(actor._object) {
		observables_t::accessor acc;

		if (_observables.find(acc, actor._registered_object)) {
			acc->second.actors.erase(&actor);
		}

		actor.set_object(nullptr, nullptr);
		return true;
	}
	return false;
}

/*****************************************************************************
 * PVHive::PVHive::unregister_observer()
 *****************************************************************************/

bool PVHive::PVHive::unregister_observer(PVObserverBase& observer)
{
	if(observer._object) {
		observables_t::accessor acc;

		if (_observables.find(acc, observer._registered_object)) {
			acc->second.observers.remove(&observer);
		}

		observer._object = nullptr;
		return true;
	}
	return false;
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
					pit->set_object(nullptr, nullptr);
				}

				// and observers
				for (auto pit = pacc->second.observers.rbegin();
				     pit != pacc->second.observers.rend(); ++pit) {
					(*pit)->object_about_to_be_unregistered();
				}
				for (auto pit = pacc->second.observers.rbegin();
				     pit != pacc->second.observers.rend(); ++pit) {
					(*pit)->about_to_be_deleted();
				}
			}
			_observables.erase(pacc);
		}

		// unregistering actors...
		for (auto it : acc->second.actors) {
			it->set_object(nullptr, nullptr);
		}

		// and observers
		for (auto it = acc->second.observers.rbegin();
		     it != acc->second.observers.rend(); ++it) {
			(*it)->object_about_to_be_unregistered();
		}
		for (auto it = acc->second.observers.rbegin();
		     it != acc->second.observers.rend(); ++it) {
			(*it)->about_to_be_deleted();
		}

		// finally, the entry of object is removed
		_observables.erase(acc);
	}
}

bool PVHive::PVHive::unregister_func_observer(PVFuncObserverBase& observer, void* f)
{
	if(observer._object) {
		observables_t::accessor acc;

		if (_observables.find(acc, observer._registered_object)) {
			func_observers_t& fobs(acc->second.func_observers);
			func_observers_t::const_iterator it_fo, it_fo_e;
			std::tie(it_fo, it_fo_e) = fobs.equal_range(f);
			func_observers_t::const_iterator it_to_del = std::find_if(it_fo, it_fo_e, [=,&observer](func_observers_t::value_type const& it) { return it.second == &observer; });
			if (it_to_del != fobs.end()) {
				fobs.erase(it_to_del);
			}
		}

		observer._object = nullptr;
		observer._registered_object = nullptr;
		return true;
	}
	return false;
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

void PVHive::PVHive::do_refresh_observers_maybe_recursive(void *object)
{
	observables_t::const_accessor acc;

	if (_observables.find(acc, object)) {
		for (auto it : acc->second.observers) {
			if (it->accept_recursive_refreshes()) {
				it->refresh();
			}
		}
	}
}

void PVHive::PVHive::do_about_to_refresh_observers(void* object)
{
	observables_t::const_accessor acc;

	if (_observables.find(acc, object)) {
		for (auto it : acc->second.observers) {
			it->about_to_be_refreshed();
		}
	}
}

void PVHive::PVHive::refresh_observers(PVCore::PVDataTreeObjectWithParentBase const* object, void* obj_refresh)
{
	// object must be a valid address
	assert(object != nullptr);

	do_refresh_observers(obj_refresh);

	// AG: in test cases (mainly PVGuiQt and PVParallelView), we use "fake" objects that have no parents,
	// thus this check is necessary!
	PVCore::PVDataTreeObjectBase const* const parent = object->get_parent_base();
	if (parent) {
		refresh_observers_maybe_recursive(parent);
	}
}

void PVHive::PVHive::refresh_observers_maybe_recursive(PVCore::PVDataTreeObjectWithParentBase const* object, void* obj_refresh)
{
	// object must be a valid address
	assert(object != nullptr);

	do_refresh_observers_maybe_recursive(obj_refresh);

	// AG: in test cases (mainly PVGuiQt and PVParallelView), we use "fake" objects that have no parents,
	// thus this check is necessary!
	PVCore::PVDataTreeObjectBase const* const parent = object->get_parent_base();
	if (parent) {
		refresh_observers_maybe_recursive(parent);
	}
}
