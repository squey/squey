/**
 * \file PVFuncObserver.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvhive/PVFuncObserver.h>

__impl::PVFuncObserverQtBase::PVFuncObserverQtBase()
{
	std::cout << "PVFuncObserverQtBase::PVFuncObserverQtBase" << std::endl;

	connect(this, SIGNAL(about_to_be_refreshed_signal(const void*)), this, SLOT(about_to_be_refreshed_slot(const void*)));
	connect(this, SIGNAL(refresh_signal(const void*)), this, SLOT(refresh_slot(const void*)));
}

void __impl::PVFuncObserverQtBase::do_about_to_be_updated_split(const void* args) const
{
	emit about_to_be_refreshed_signal(args);
}

void __impl::PVFuncObserverQtBase::do_update_split(const void* args) const
{
	emit refresh_signal(args);
}

void __impl::PVFuncObserverQtBase::about_to_be_refreshed_slot(const void* args) const
{
	about_to_be_updated_cast(args);
}

void __impl::PVFuncObserverQtBase::refresh_slot(const void* args) const
{
	update_cast(args);
}

