
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <map>

#include <boost/tuple/tuple.hpp>

#include <pvhive/PVObserver.h>

namespace PVHive
{

class PVActorBase;

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
	void register_actor(T& p, PVActor<T>& actor)
	{
		_actors.insert(std::make_pair((void*) &p, (PVActorBase*) &actor));
		actor._object = &p;
	}

	/**
	 * Register an observer for an address
	 *
	 * @param p the observed address
	 * @param observer the observer
	 */
	template <class T>
	void register_observer(T const& p, PVObserver<T>& observer)
	{
		_observers.insert(std::make_pair((void*) &p, (PVObserverBase*) &observer));

		// an observer must be set for only one object
		assert(observer._object == nullptr);

		observer._object = &p;
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
		observers_t::const_iterator it,it_end;
		boost::tie(it,it_end) = _observers.equal_range((void*) obj);
		for (; it != it_end; it++) {
			it->second->refresh();
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
};

}

#endif // LIBPVHIVE_PVHIVE_H
