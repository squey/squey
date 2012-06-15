
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <map>
#include <functional>

#include <QThread>

#include <boost/thread.hpp>

#include <pvhive/PVObserver.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

namespace __impl
{

typedef std::function<void()> function_t;

}

template <class T>
class PVActor;

/**
 * \attention any specialization of template method ::call_object() *must* be
 * declared the namespace PVHive (you can use the macros
 * PVHIVE_CALL_OBJECT_BLOCK_BEGIN() and PVHIVE_CALL_OBJECT_BLOCK_END()).
*/

#define PVHIVE_CALL_OBJECT_BLOCK_BEGIN() namespace PVHive {
#define PVHIVE_CALL_OBJECT_BLOCK_END() }

class PVHive : public QThread
{
	Q_OBJECT

public:
	/**
	 * @return a reference on the global PVHive
	 */
	static PVHive &get()
	{
		if (_hive == nullptr) {
			_hive = new PVHive;
		}
		return *_hive;
	}

public:
	/**
	 * Register an actor for an address
	 *
	 * @param p the managed address
	 * @param actor the actor
	 */
	template <class T>
	void register_actor(T& object, PVActorBase& actor)
	{
		{
			boost::lock_guard<boost::mutex> lock(_actors_mutex);
			_actors.insert(std::make_pair((void*) &object, &actor));
		}

		// an actor must be set for only one object
		assert(actor._object == nullptr);

		actor._object = (void*) &object;
	}

	/**
	 * Helper method easily create and register an actor for an address
	 *
	 * @param p the observed address
	 * @return the actor
	 */
	template <class T>
	PVActor<T>* register_actor(T& object)
	{
		PVActor<T>* actor = new PVActor<T>();
		register_actor(object, *actor);

		return actor;
	}

	/**
	 * Unregister an actor an notify all observers of its managed address
	 * that it is about to be deleted.
	 *
	 */
	void unregister_actor(PVActorBase& actor);

	/**
	 * Register an observer for an address
	 *
	 * @param p the observed address
	 * @param observer the observer
	 */
	template <class T>
	void register_observer(T const& object, PVObserverBase& observer)
	{
		{
			write_lock_t write_lock(_observers_lock);
			_observers.insert(std::make_pair((void*) &object, &observer));
		}

		// an observer must be set for only one object
		assert(observer._object == nullptr);

		observer._object = (void*) &object;
	}

	/**
	 * Unregister an observer.
	 *
	 */
	void unregister_observer(PVObserverBase& observer);

public:
	/**
	 * Generic call to apply an action on a object
	 *
	 * @param obj the managed address
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object(T* object, P... params)
	{
		call_object_default<T, F, f>(object, params...);
	}

	/**
	 * Tell all observers of an address that a change has occurred
	 *
	 * @param object the observed address
	 */
	template <typename T>
	void refresh_observers(T const* object)
	{
		emit refresh_observers((void*)object);
	}

private:
	/**
	 * Apply an action on a object and propagate the change event
	 *
	 * @param object the managed address
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object_default(T* obj, P... params)
	{
		emit invoke_object(std::bind(f, obj, params...));
		emit refresh_observers((void*)obj);
	}

private:
	PVHive(QObject *parent = nullptr);
	void run();

signals:
	void invoke_object(__impl::function_t func);

	void refresh_observers(void *object);

private slots:
	void do_invoke_object(__impl::function_t func);

	void do_refresh_observers(void *object);

private:
	static PVHive *_hive;

	typedef std::multimap<void*, PVActorBase*> actors_t;
	typedef std::multimap<void*, PVObserverBase*> observers_t;
	actors_t       _actors;
	observers_t    _observers;

	// thread safety
	typedef boost::unique_lock<boost::shared_mutex> write_lock_t;
	typedef boost::shared_lock<boost::shared_mutex> read_lock_t;
	boost::shared_mutex _observers_lock;
	boost::mutex _actors_mutex;
};

}

#endif // LIBPVHIVE_PVHIVE_H
