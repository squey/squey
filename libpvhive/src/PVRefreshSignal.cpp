
#include <pvhive/PVRefreshSignal.h>

void PVHive::__impl::PVRefreshSignal::do_refresh_signal(PVObserverBase* o)
{
	QMetaObject::invokeMethod(_refresh_object, _refresh_func, Qt::AutoConnection,
	                          Q_ARG(PVHive::PVObserverBase*, o));
	_refresh_sem.release(1);
}


void PVHive::__impl::PVRefreshSignal::do_atbd_signal(PVObserverBase* o)
{
	QMetaObject::invokeMethod(_atbd_object, _atbd_func, Qt::AutoConnection,
	                          Q_ARG(PVHive::PVObserverBase*, o));
	_atbd_sem.release(1);
}
