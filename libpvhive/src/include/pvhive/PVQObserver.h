
#ifndef LIBPVHIVE_PVQOBSERVER_H
#define LIBPVHIVE_PVQOBSERVER_H

#include <QObject>

#include <pvhive/PVObserver.h>
#include <pvhive/PVQRefresh.h>

namespace PVHive
{

template <class T>
class PVQObserver : public __impl::PVQRefresh, public PVObserver<T>
{
public:
	PVQObserver(QObject* parent) :
		__impl::PVQRefresh(parent)
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

#endif // LIBPVHIVE_PVQOBSERVER_H
