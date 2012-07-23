/**
 * \file PVActorBase.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVActorBase.h>
#include <pvhive/PVHive.h>

PVHive::PVActorBase::~PVActorBase()
{
	PVHive::get().unregister_actor(*this);
}
