
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <map>

#include <boost/thread.hpp>

#include <pvhive/PVObserver.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

template <class T>
class PVActor;

class PVHive
{
	/* a template can not use a pointer on template class; so we have
	 * PVActorBase and PVObserverBase
	 */
	typedef std::multimap<void*, PVActorBase*> actors_t;
	typedef std::multimap<void*, PVObserverBase*> observers_t;

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
	 * Unregister an actor an notify all observers of its manager address
	 * that it is about to be deleted.
	 *
	 */
	void unregister_actor(PVActorBase& actor)
	{
		{
			read_lock_t read_lock(_observers_lock);
			auto ret = const_cast<observers_t&>(_observers).equal_range(actor._object);
			for (auto it = ret.first; it != ret.second; ++it) {
				it->second->about_to_be_deleted();
			}
		}
		{
			boost::lock_guard<boost::mutex> lock(_actors_mutex);
			_actors.erase(actor._object);
			actor._object = nullptr;
		}
	}

	/**
	 * Register an observer for an address
	 *
	 * @param p the observed address
	 * @param observer the observer
	 */
	template <class T>
	void register_observer(T const& p, PVObserverBase& observer)
	{
		{
			write_lock_t write_lock(_observers_lock);
			_observers.insert(std::make_pair((void*) &p, &observer));
		}

		// an observer must be set for only one object
		assert(observer._object == nullptr);

		observer._object = (void*) &p;
	}

	/**
	 * Unregister an observer.
	 *
	 */
	void unregister_observer(PVObserverBase& observer)
	{
		{
			write_lock_t write_lock(_observers_lock);
			_observers.erase(observer._object);
			observer._object = nullptr;
		}
	}

public:
	/**
	 * Generic call to apply an action on a object
	 *
	 * @param obj the managed address
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object(T* obj, P... params)
	{
		call_object_default<T, F, f>(obj, params...);
	}

	/**
	 * tells all observers of an address that a change has occurred
	 *
	 * @param obj the observed address
	 */
	template <typename T>
	void refresh_observers(T const* obj)
	{
		{
			read_lock_t read_lock(_observers_lock);
			auto ret = const_cast<observers_t&>(_observers).equal_range((void*) obj);
			for (auto it = ret.first; it != ret.second; ++it) {
				it->second->refresh();
			}
		}
	}

private:
	/**
	 * Apply an action on a object and propagate the change event
	 *
	 * @param obj the managed address
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object_default(T* obj, P... params)
	{
		(obj->*f)(params...);
		refresh_observers(obj);
	}

private:
	static PVHive *_hive;
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
