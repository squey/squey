/**
 * \file PVObserverCallback.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LIBPVHIVE_PVOBSERVERCALLBACK_H
#define LIBPVHIVE_PVOBSERVERCALLBACK_H

#include <pvhive/PVObserver.h>

namespace PVHive
{

template <class T, class RefreshF, class DeleteF>
class PVObserverCallback : public PVObserver<T>
{
public:
	PVObserverCallback(RefreshF const& r, DeleteF const& d) :
		_refresh_cb(r),
		_delete_cb(d)
	{}

protected:
	virtual void refresh()
	{

		_refresh_cb(PVObserver<T>::get_object());
	}

	virtual void about_to_be_deleted()
	{
		_delete_cb(PVObserver<T>::get_object());
	}

private:
	RefreshF _refresh_cb;
	DeleteF  _delete_cb;
};

template <class T, class RefreshF, class DeleteF>
PVObserverCallback<T, RefreshF, DeleteF>
create_observer_callback(RefreshF const& r, DeleteF const& d)
{
	return PVObserverCallback<T, RefreshF, DeleteF>(r, d);
}

}

#endif // LIBPVHIVE_PVOBSERVERCALLBACK_H
