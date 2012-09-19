/**
 * \file PVFuncObserver.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVFuncObserver.h>

__impl::PVFuncObserverSignalBase::PVFuncObserverSignalBase(void* f):
	PVHive::PVFuncObserverBase(f)
{
	connect(this, SIGNAL(about_to_be_refreshed_signal(const void*)), this, SLOT(about_to_be_refreshed_slot(const void*)));
	connect(this, SIGNAL(refresh_signal(const void*)), this, SLOT(refresh_slot(const void*)));
}

void __impl::PVFuncObserverSignalBase::do_about_to_be_updated_impl(const void* args) const
{

	emit about_to_be_refreshed_signal(args);
}

void __impl::PVFuncObserverSignalBase::do_update_impl(const void* args) const
{
	emit refresh_signal(args);
}

void __impl::PVFuncObserverSignalBase::about_to_be_refreshed_slot(const void* args) const
{
	call_about_to_be_updated_with_casted_args(args);
}

void __impl::PVFuncObserverSignalBase::refresh_slot(const void* args) const
{
	call_update_with_casted_args(args);
}

