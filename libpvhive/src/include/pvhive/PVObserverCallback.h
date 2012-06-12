
#ifndef LIBPVHIVE_PVOBSERVERCALLBACK_H
#define LIBPVHIVE_PVOBSERVERCALLBACK_H

#incldue <pvhive/PVObserver.h>

namespace PVHive
{

template <class T, class RefreshF, class DeleteF>
class ObserverCallback : public Observer<T>
{
public:
	ObserverCallback(RefreshF const& r, DeleteF const& d):
		_refresh_cb(r),
		_delete_cb(d)
	{}

protected:
	virtual void refresh()
	{

		_refresh_cb(Observer<T>::get_object());
	}

	virtual void about_to_be_deleted()
	{
		_delete_cb(Observer<T>::get_obj());
	}

private:
	RefreshF _refresh_cb;
	DeleteF  _delete_cb;
};

template <class T, class RefreshF, class DeleteF>
ObserverCallback<T, RefreshF, DeleteF>
create_observer_callback(RefreshF const& r, DeleteF const& d)
{
	return ObserverCallback<T, RefreshF, DeleteF>(r, d);
}

}

#endif // LIBPVHIVE_PVOBSERVERCALLBACK_H
