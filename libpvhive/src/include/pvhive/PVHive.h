
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <iostream>
#include <set>
#include <exception>

#include <tbb/concurrent_hash_map.h>

#include <pvhive/PVObserver.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

class no_object : public std::exception
{
public:
	no_object()
	{}

	virtual ~no_object() throw()
	{}

	virtual const char * what() const throw()
	{
		return "registering an actor for an unknown object";
	}
};

template <class T>
class PVActor;

/**
 * @class PVHive
 *
 * @brief The PVHive a system to ease update propagation in programs with
 * interdependant objects.
 *
 * When dealing with changing objects, keeping all dependants objects updated
 * can be complex and can need lots of lines of code to maintain the coherency;
 * as keeping a correct behaviour each time a new functionnality is added.
 *
 * The PVHive also implements a callback-like system to automatize this
 * action-reaction scheme. It is based on 3 concepts:
 * - the hive which is the keeper of list of callbacks for each observable
 *   objects;
 * - the actor which subscribe to notify updates when modifying a given object;
 * - the observer which subscribe to update notification for a given object.
 *
 * Note that:
 * - an actor can act on an object which have no observer (as no opened view
 *   for a data set);
 * - an observer can be registered for an object which have no actor (as no
 *   opened property editor for a property).
 *
 * @attention any specialization of template method ::call_object() *must* be
 * declared in the namespace PVHive (you can use the macros
 * PVHIVE_CALL_OBJECT_BLOCK_BEGIN() and PVHIVE_CALL_OBJECT_BLOCK_END()).
 */

#define PVHIVE_CALL_OBJECT_BLOCK_BEGIN() namespace PVHive {
#define PVHIVE_CALL_OBJECT_BLOCK_END() }

class PVHive
{
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
	 * Register an object
	 *
	 * @param object the new managed object
	 */
	template <class T>
	void register_object(T& object)
	{
		observables_t::accessor acc;

		_observables.insert(acc, (void*) &object);
	}

	/**
	 * Register a member variable of an object
	 *
	 * @param object the parent object
	 * @param prop_get: a function returning a reference on object's property
	 *
	 * @attention using a method as prop_get will not compile.
	 */
	template <class T, class F>
	void register_object(T const& object, F const &prop_get)
	{
		auto &property = prop_get(object);

		observables_t::accessor acc;

		// adding property's entry
		_observables.insert(acc, (void*) &property);

		// create/get object's entry
		_observables.insert(acc, (void*) &object);
		acc->second.properties.insert((void*) &property);
	}

	/**
	 * Register an actor for an obect
	 *
	 * @param object the managed object
	 * @param actor the actor
	 */
	template <class T>
	void register_actor(T& object, PVActorBase& actor)
	{
		// an actor must be set for only one object
		assert(actor.get_object() == nullptr);

		observables_t::accessor acc;

		if (_observables.find(acc, (void*) &object) == false) {
			throw no_object();
		}

		// create/get object's entry
		acc->second.actors.insert(&actor);
		actor.set_object((void*) &object);
	}

	/**
	 * Helper method easily create and register an actor for an object
	 *
	 * @param object the observed object
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
	 * Register an observer for an object
	 *
	 * @param object the observed object
	 * @param observer the observer
	 */
	template <class T>
	void register_observer(T const& object, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer._object == nullptr);

		observables_t::accessor acc;

		// create/get object's entry
		_observables.insert(acc, (void*) &object);
		acc->second.observers.insert(&observer);
		observer._object = (void*) &object;
	}

	/**
	 * Register an observer for a member variable of an object
	 *
	 * @param object the parent object
	 * @param prop_get: a function returning a reference on object's property
	 * @param observer the observer
	 *
	 * @attention using a method as prop_get will not compile.
	 */
	template <class T, class F>
	void register_observer(T const& object, F const &prop_get, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer._object == nullptr);

		auto &property = prop_get(object);

		observables_t::accessor acc;

		// adding observer
		// create/get property's entry
		_observables.insert(acc, (void*) &property);
		acc->second.observers.insert(&observer);
		observer._object = (void*) &property;

		// adding property
		// create/get object's entry
		_observables.insert(acc, (void*) &object);
		acc->second.properties.insert((void*) &property);
	}

	/**
	 * Unregister an object
	 *
	 * @param object the object
	 */
	template <typename T>
	void unregister_object(T const &object)
	{
		// if T is a pointer, its address is used, not its value
		static_assert(!std::is_pointer<T>::value, "PVHive::PVHive::unregister_object<T>(T const &) does not accept pointer as parameter");

		unregister_object((void*)&object);
	}

	/**
	 * Unregister an actor an notify all dependent observers
	 * that they must stop observing
	 *
	 * @param actor the actor
	 */
	void unregister_actor(PVActorBase& actor);

	/**
	 * Unregister an observer
	 *
	 * @param observer the observer
	 */
	void unregister_observer(PVObserverBase& observer);

private:
	/**
	 * Unregister an object
	 *
	 * @param object the object
	 */
	void unregister_object(void *object);

public:
	/**
	 * Generic call to apply an action on a object
	 *
	 * @param object the managed object
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object(T* object, P... params)
	{
		// object must be a valid address
		assert(object != nullptr);

		call_object_default<T, F, f>(object, params...);
	}

	/**
	 * Tell all observers of an object that a change has occurred
	 *
	 * @param object the observed object
	 */
	template <typename T>
	void refresh_observers(T const* object)
	{
		// object must be a valid address
		assert(object != nullptr);

		do_refresh_observers((void*)object);
	}

	void print() const
	{
		std::cout << "PVHive - " << this << " - content:" << std::endl;
		for (auto it : _observables) {
			std::cout << "    " << it.first << std::endl;

			std::cout << "        actors:" << std::endl;
			for (auto it2 : it.second.actors) {
				std::cout << "            " << it2 << std::endl;
			}

			std::cout << "        observers:" << std::endl;
			for (auto it2 : it.second.observers) {
				std::cout << "            " << it2 << std::endl;
			}

			std::cout << "        properties:" << std::endl;
			for (auto it2 : it.second.properties) {
				std::cout << "            " << it2 << std::endl;
			}
		}

	}

private:
	/**
	 * Apply an action on a object and propagate the change event
	 *
	 * @param object the managed object
	 * @param params the method parameters
	 */
	template <typename T, typename F, F f, typename... P>
	void call_object_default(T* object, P... params)
	{
		// object must be a valid address
		assert(object != nullptr);

		(object->*f)(params...);
		do_refresh_observers((void*)object);
	}

	void do_refresh_observers(void *object);

private:
	PVHive() {}
	~PVHive() {}
	PVHive(const PVHive&) {}
	PVHive &operator=(const PVHive&) { return *this; }

private:
	static PVHive *_hive;

	typedef std::set<PVActorBase*> actors_t;
	typedef std::set<PVObserverBase*> observers_t;
	typedef std::set<void*> properties_t;

	struct observable_t
	{
		bool empty() const
		{
			return actors.empty() && observers.empty() && properties.empty();
		}

		actors_t  actors;
		observers_t  observers;
		properties_t properties;
	};

	typedef tbb::concurrent_hash_map<void*, observable_t > observables_t;
	observables_t _observables;
};

}

#endif // LIBPVHIVE_PVHIVE_H
