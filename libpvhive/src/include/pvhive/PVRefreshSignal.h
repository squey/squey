
#ifndef LIBPVHIVE_PVREFRESHSIGNAL_H
#define LIBPVHIVE_PVREFRESHSIGNAL_H

#include <QMetaType>
#include <QObject>
#include <QSemaphore>

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
		_refresh_sem(0),
		_atbd_sem(0)
	{}

public:
	inline void connect_refresh(QObject *receiver, const char *func)
	{
		_refresh_object = receiver;
		_refresh_func = func;
		connect(this, SIGNAL(refresh_signal(PVHive::PVObserverBase*)),
		        this, SLOT(do_refresh_signal(PVHive::PVObserverBase*)));
	}

	inline void connect_about_to_be_deleted(QObject* receiver, const char *func)
	{
		_atbd_object = receiver;
		_atbd_func = func;
		connect(this, SIGNAL(about_to_be_deleted_signal(PVHive::PVObserverBase*)),
		        this, SLOT(do_atbd_signal(PVHive::PVObserverBase*)));
	}

protected:
	inline void emit_refresh_signal(PVObserverBase* o)
	{
		emit refresh_signal(o);
		_refresh_sem.acquire(1);
	}

	inline void emit_about_to_be_deleted_signal(PVObserverBase *o)
	{
		emit about_to_be_deleted_signal(o);
		_atbd_sem.acquire(1);
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
#else
	void refresh_signal(PVObserverBase *o);
	void about_to_be_deleted_signal(PVObserverBase *o);
#endif

private slots:
/* same problem about Qt and namespaces
 */
#ifdef Q_MOC_RUN
	void do_refresh_signal(PVHive::PVObserverBase* o);
	void do_atbd_signal(PVHive::PVObserverBase* o);
#else
	void do_refresh_signal(PVObserverBase* o);
	void do_atbd_signal(PVObserverBase* o);
#endif

private:
	QObject    *_refresh_object;
	const char *_refresh_func;
	QSemaphore  _refresh_sem;

	QObject   *_atbd_object;
	const char*_atbd_func;
	QSemaphore _atbd_sem;
};

}

}

#endif // LIBPVHIVE_PVREFRESHSIGNAL_H
