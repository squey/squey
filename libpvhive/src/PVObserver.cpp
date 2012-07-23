/**
 * \file PVObserver.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVObserver.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVHive.h>

PVHive::PVObserverBase::~PVObserverBase()
{
	if (_object != nullptr) {
		PVHive::get().unregister_observer(*this);
	}
}

PVHive::PVFuncObserverBase::~PVFuncObserverBase()
{
	if (_object != nullptr) {
		PVHive::get().unregister_func_observer(*this, _f);
	}
}
