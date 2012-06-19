
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
#if 0
	// code for asynchronous call_object/refresh()
	/* Qt has to know the type function_t for signals/slots; otherwise
	 * there is the following error at run-time:
	 * QObject::connect: Cannot queue arguments of type '__impl::function_t'
	 * (Make sure '__impl::function_t' is registered using qRegisterMetaType().)
	 *
	 * This problem occurs when a non QThread thread do an action.
	 */
	qRegisterMetaType<__impl::function_t>("__impl::function_t");

	connect(this, SIGNAL(invoke_object(__impl::function_t)),
	        this, SLOT(do_invoke_object(__impl::function_t)));

	connect(this, SIGNAL(refresh_observers(void*)),
	        this, SLOT(do_refresh_observers(void*)));

	start();
#endif
}

/*****************************************************************************
 * PVHive::PVHive::PVHive()
 *****************************************************************************/

PVHive::PVHive::PVHive(const PVHive&) :
	QThread(nullptr)
{}


/*****************************************************************************
 * PVHive::PVHive::PVHive()
 *****************************************************************************/
PVHive::PVHive &PVHive::PVHive::operator=(const PVHive&)
{
	return *this;
}

/*****************************************************************************
 * PVHive::PVHive::run()
 *****************************************************************************/
PVHive::PVHive::~PVHive()
{
	quit();
	wait();
}

/*****************************************************************************
 * PVHive::PVHive::run()
 *****************************************************************************/

void PVHive::PVHive::run()
{
	exec();
}

/*****************************************************************************
 * PVHive::PVHive::unregister_actor()
 *****************************************************************************/

void PVHive::PVHive::unregister_actor(PVActorBase& actor)
{
	unregister_object(actor._object);

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
 * PVHive::PVHive::unregister_object()
 *****************************************************************************/

void PVHive::PVHive::unregister_object(void *object)
{
	// the object must be a valid address
	assert(object != nullptr);

	emit_about_to_be_deleted(object);
}

/*****************************************************************************
 * PVHive::PVHive::emit_about_to_be_deleted()
 *****************************************************************************/

void PVHive::PVHive::emit_about_to_be_deleted(void* object)
{
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
