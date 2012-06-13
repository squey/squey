
#ifndef LIBPVHIVE_PVOBSERVERSIGNAL_H
#define LIBPVHIVE_PVOBSERVERSIGNAL_H

#include <QObject>

#include <pvhive/PVObserver.h>
#include <pvhive/PVRefreshSignal.h>

namespace PVHive
{

template <class T>
class PVObserverSignal : public __impl::PVRefreshSignal, public PVObserver<T>
{
public:
	PVObserverSignal(QObject* parent) :
		__impl::PVRefreshSignal(parent)
	{}

protected:
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
