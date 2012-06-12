
#ifndef LIBPVHIVE_PVREFRESHSIGNAL_H
#define LIBPVHIVE_PVREFRESHSIGNAL_H

#include <QObject>

namespace PVHive
{

class PVObserverBase;

namespace __impl
{

class PVRefreshSignal : public QObject
{
	Q_OBJECT

public:
	PVRefreshSignal(QObject *parent = nullptr) :
		QObject(parent)
	{}

public:
	inline void connect_refresh(QObject *receiver, const char *slot)
	{
		connect(this, SIGNAL(refresh_signal(PVObserverBase*)), receiver, slot);
	}

	inline void connect_about_to_be_deleted(QObject* receiver, const char *slot)
	{
		connect(this, SIGNAL(about_to_be_deleted_signal(PVObserverBase*)), receiver, slot);
	}

protected:
	inline void emit_refresh_signal(PVObserverBase* o)
	{
		emit refresh_signal(o);
	}

	inline void emit_about_to_be_deleted_signal(PVObserverBase *o)
	{
		emit about_to_be_deleted_signal(o);
	}

signals:
	void refresh_signal(PVObserverBase *o);
	void about_to_be_deleted_signal(PVObserverBase *o);
};

}

}

#endif // LIBPVHIVE_PVREFRESHSIGNAL_H
