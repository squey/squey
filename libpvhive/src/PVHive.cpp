
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
 * PVHive::PVHive::unregister_actor()
 *****************************************************************************/

void PVHive::PVHive::unregister_actor(PVActorBase& actor)
{
	// the actor must have a valid object
	assert(actor._object != nullptr);

	emit_about_to_be_deleted(actor._object);

	boost::lock_guard<boost::mutex> lock(_actors_mutex);
	_actors.erase(actor._object);
	actor._object = nullptr;
}

/*****************************************************************************
 * PVHive::PVHive::unregister_observer()
 *****************************************************************************/

void PVHive::PVHive::unregister_observer(PVObserverBase& observer)
{
	// the observer must have a valid object
	assert(observer._object != nullptr);

	write_lock_t write_lock(_observers_lock);

	_observers.erase(observer._object);
	observer._object = nullptr;
}

/*****************************************************************************
 * PVHive::PVHive::emit_about_to_be_deleted()
 *****************************************************************************/

void PVHive::PVHive::emit_about_to_be_deleted(void* object)
{
	// object must be a valid address
	assert(object != nullptr);

	read_lock_t read_lock(_observers_lock);
	auto ret = const_cast<observers_t&>(_observers).equal_range(object);
	for (auto it = ret.first; it != ret.second; ++it) {
		it->second->about_to_be_deleted();
	}
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
