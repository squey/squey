
#ifndef LIBPVHIVE_PVOBSERVERCALLBACK_H
#define LIBPVHIVE_PVOBSERVERCALLBACK_H

#include <pvhive/PVObserver.h>

namespace PVHive
{

/**
 * @class PVObserverCallback
 *
 * A template class to specify observers using functions instead
 * of methods.
 *
 * Used functions can be of any kind: lambda, static defined,
 * std::function, boost::function, etc.
 *
 * All subclasses must implements PVObserverBase::refresh() and
 * PVObserverBase::about_to_be_deleted().
 */
template <class T, class RefreshF, class DeleteF>
class PVObserverCallback : public PVObserver<T>
{
public:
	PVObserverCallback()
	{}

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

/**
 * Helper function a create a PVObserverCallback without dealing
 * with the PVObserverCallback class itself.
 */
template <class T, class RefreshF, class DeleteF>
PVObserverCallback<T, RefreshF, DeleteF>
create_observer_callback(RefreshF const& r, DeleteF const& d)
{
	return PVObserverCallback<T, RefreshF, DeleteF>(r, d);
}

}

#endif // LIBPVHIVE_PVOBSERVERCALLBACK_H
