/**
 * \file PVObserver.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVObserver.h>
#include <pvhive/PVHive.h>

PVHive::PVObserverBase::~PVObserverBase()
{
	PVHive::get().unregister_observer(*this);
}
