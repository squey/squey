
#ifdef DEBUG
#include <iostream>
#endif

#include <QMetaType>

#include <pvhive/PVHive.h>
#include <pvhive/PVActorBase.h>
#include <pvhive/PVObserver.h>

/*****************************************************************************/

PVHive::PVHive *PVHive::PVHive::_hive = nullptr;

/*****************************************************************************
 * PVHive::PVHive::PVHive()
 *****************************************************************************/

PVHive::PVHive::PVHive(QObject *parent) :
	QThread(parent)
{
	// Qt has to know the type function_t for signals/slots
	qRegisterMetaType<__impl::function_t>("__impl::function_t");

	connect(this, SIGNAL(invoke_object(__impl::function_t)),
	        this, SLOT(do_invoke_object(__impl::function_t)));

	connect(this, SIGNAL(refresh_observers(void*)),
	        this, SLOT(do_refresh_observers(void*)));

	start();
}

/*****************************************************************************
 * PVHive::PVHive::run()
 *****************************************************************************/

void PVHive::PVHive::run()
{
#ifdef DEBUG
	std::cout << "PVHive::PVHive::thread is " << thread() << std::endl;
#endif
	exec();
}

/*****************************************************************************
 * PVHive::PVHive::do_invoke_object()
 *****************************************************************************/

void PVHive::PVHive::do_invoke_object(__impl::function_t func)
{
	func();
}

/*****************************************************************************
 * PVHive::PVHive::do_refresh_observers()
 *****************************************************************************/

void PVHive::PVHive::do_refresh_observers(void *object)
{
	auto ret = const_cast<observers_t&>(_observers).equal_range(object);
	for (auto it = ret.first; it != ret.second; ++it) {
		it->second->refresh();
	}
}
