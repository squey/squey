
#ifndef LIBPVHIVE_PVREFRESHSIGNAL_H
#define LIBPVHIVE_PVREFRESHSIGNAL_H

#include <QObject>

namespace PVHive
{

class PVObserverBase;

}

/* Qt's signals/slots mechanism can not work properly with namespaces; leading
 * to run-time errors of type "Incompatible sender/receiver arguments" or
 * "No such signal": the signals use implicit namespaces prefix (otherwise it
 * does not compile) and the slots use explicit namespaces prefix. So that the
 * MOC's internal strcmp fails when comparing signals/slots signatures.
 *
 * Adding "using namespace NAMESPACE" solves the problem but it is error
 * prone.
 *
 * Adding the new derived class in the same namespace is not acceptable.
 *
 * So here is a hack to have PVRefreshSignal defined in its namespace *and*
 * the signals defined outside any namespace: use a (very) long and unusual
 * class name for the class definition and use "typedef" in the namespace.
 *
 * The new signals/slots mechanism of Qt 5 may solve this issue.
 */
class Hack_QtSignalsWithNoNameSpaces_PVRefreshSignal : public QObject
{
	Q_OBJECT

public:
	Hack_QtSignalsWithNoNameSpaces_PVRefreshSignal(QObject *parent = nullptr) :
		QObject(parent)
	{}

public:
	inline void connect_refresh(QObject *receiver, const char *slot)
	{
		connect(this, SIGNAL(refresh_signal(PVHive::PVObserverBase*)), receiver, slot);
	}

	inline void connect_about_to_be_deleted(QObject* receiver, const char *slot)
	{
		connect(this, SIGNAL(about_to_be_deleted_signal(PVHive::PVObserverBase*)), receiver, slot);
	}

protected:
	inline void emit_refresh_signal(PVHive::PVObserverBase* o)
	{
		emit refresh_signal(o);
	}

	inline void emit_about_to_be_deleted_signal(PVHive::PVObserverBase *o)
	{
		emit about_to_be_deleted_signal(o);
	}

signals:
	void refresh_signal(PVHive::PVObserverBase *o);
	void about_to_be_deleted_signal(PVHive::PVObserverBase *o);
};


namespace PVHive
{

namespace __impl
{

	typedef Hack_QtSignalsWithNoNameSpaces_PVRefreshSignal PVRefreshSignal;

}

}

#endif // LIBPVHIVE_PVREFRESHSIGNAL_H
