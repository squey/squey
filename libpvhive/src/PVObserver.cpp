/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvhive/PVObserver.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVHive.h>

PVHive::PVObserverBase::~PVObserverBase()
{
	if (_object != nullptr && !is_object_about_to_be_unregistered()) {
		PVHive::get().unregister_observer(*this);
	}
}

PVHive::PVFuncObserverBase::~PVFuncObserverBase()
{
	if (_object != nullptr && !is_object_about_to_be_unregistered()) {
		PVHive::get().unregister_func_observer(*this, _f);
	}
}
