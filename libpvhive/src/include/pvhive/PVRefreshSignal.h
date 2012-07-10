
#ifndef LIBPVHIVE_PVREFRESHSIGNAL_H
#define LIBPVHIVE_PVREFRESHSIGNAL_H

#include <QObject>
#include <QSemaphore>
#include <QThread>

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
		QObject(parent),
		_atbd_sem(0)
	{}

public:
	inline void connect_refresh(QObject *receiver, const char *slot)
	{
		connect(this, SIGNAL(refresh_signal(PVHive::PVObserverBase*)),
		        receiver, slot);
	}

	inline void connect_about_to_be_deleted(QObject* receiver, const char *slot)
	{
		connect(this, SIGNAL(about_to_be_deleted_signal(PVHive::PVObserverBase*)),
		        receiver, slot);
		connect(this, SIGNAL(sync_about_to_be_deleted_signal(PVHive::PVObserverBase*)),
		        this, SLOT(do_sync_atbd_signal(PVHive::PVObserverBase*)));
	}

protected:
	inline void emit_refresh_signal(PVObserverBase* o)
	{
		emit refresh_signal(o);
	}

	inline void emit_about_to_be_deleted_signal(PVObserverBase *o)
	{
		if (thread() == QThread::currentThread()) {
			// the signal will be synchronous
			emit about_to_be_deleted_signal(o);
		} else {
			// the signal will be asynchronous
			emit sync_about_to_be_deleted_signal(o);
			_atbd_sem.acquire(1);
		}
	}

signals:
/* Qt's signals/slots mechanism can not work properly with namespaces; leading
 * to run-time errors of type "Incompatible sender/receiver arguments" or
 * "No such signal": the signals use implicit namespaces prefix (otherwise it
 * does not compile) and the slots use explicit namespaces prefix. So that the
 * MOC's internal strcmp fails when comparing signals/slots signatures.
 *
 * To get round, the symbol Q_MOC_RUN has to be used to test if moc is running
 * or not. See http://qt-project.org/doc/qt-4.8/moc.html
 */
#ifdef Q_MOC_RUN
	void refresh_signal(PVHive::PVObserverBase *o);
	void about_to_be_deleted_signal(PVHive::PVObserverBase *o);
	void sync_about_to_be_deleted_signal(PVHive::PVObserverBase *o);
#else
	void refresh_signal(PVObserverBase *o);
	void about_to_be_deleted_signal(PVObserverBase *o);
	void sync_about_to_be_deleted_signal(PVObserverBase *o);
#endif

private slots:
/* same problem about Qt and namespaces
 */
#ifdef Q_MOC_RUN
	void do_sync_atbd_signal(PVHive::PVObserverBase* o);
#else
	void do_sync_atbd_signal(PVObserverBase* o);
#endif

private:
	QSemaphore _atbd_sem;
};

}

}

#endif // LIBPVHIVE_PVREFRESHSIGNAL_H
