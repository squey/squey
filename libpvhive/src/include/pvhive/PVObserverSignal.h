/**
 * \file PVObserverSignal.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LIBPVHIVE_PVOBSERVERSIGNAL_H
#define LIBPVHIVE_PVOBSERVERSIGNAL_H

#include <QObject>

#include <pvhive/PVObserver.h>
#include <pvhive/PVRefreshSignal.h>

namespace PVHive
{

/**
 * @class PVObserverSignal
 *
 * A template class to specify observers which use Qt's signal/slot
 * mechanism.
 *
 * To connect slot to the events, use the following methods (defined in
 * PVRefreshSignal.h:
 * - for "about_to_be_refreshed" event:
 *   void connect_about_to_be_refreshed(QObject *receiver, const char *slot);
 * - for "refresh" event:
 *   void connect_refresh(QObject *receiver, const char *slot);
 * - for "about_to_be_deleted" event:
 *   void connect_about_to_be_deleted(QObject* receiver, const char *slot);
 */
template <class T>
class PVObserverSignal : public __impl::PVRefreshSignal, public PVObserver<T>
{
public:
	PVObserverSignal(QObject* parent = NULL) :
		__impl::PVRefreshSignal(parent),
		PVObserver<T>()
	{}

protected:
	virtual void about_to_be_refreshed()
	{
		emit_about_to_be_refreshed_signal(this);
	}

	virtual void refresh()
	{
		emit_refresh_signal(this);
	}

	virtual void about_to_be_deleted()
	{
		emit_about_to_be_deleted_signal(this);
	}
};

}

#endif // LIBPVHIVE_PVOBSERVERSIGNAL_H
