
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <iostream>
#include <set>
#include <list>
#include <algorithm>
#include <exception>

#include <tbb/concurrent_hash_map.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVObserver.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

/**
 * \addtogroup exceptions
 * @{
 * Thrown when trying to register an actor on an unregistered object.
 */
class no_object : public std::exception
{
public:
	no_object(const char* text) : _text(text)
	{}

	virtual ~no_object() throw()
	{}

	virtual const char * what() const throw()
	{
		return _text;
	}
private:
	const char *_text;
};

/// @}

template <class T>
class PVActor;

namespace __impl
{

// declaration of hive_deleter
template <typename T>
inline void hive_deleter(T *ptr);

}

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
 * To ease objects deletion, each registered object is a shared pointer whose
 * deleter is set by the PVHive to automatically unregistered it.
 *
 * There are 2 kinds of notifications:
 * - "refresh": when an object is modified;
 * - "about_to_be_deleted" when an object is unregistered (ie. it will be deleted)
 *   the object is still usable until this callback returns.
 *
 * Note that:
 * - an actor can act on an object which have no observer (as no opened view
 *   for a data set);
 * - an observer can be registered for an object which have no actor (as no
 *   opened property editor for a property).
 *
 * Experimental (and theorical) memory usage:
 * - hive    : 568 octets
 * - object  : 195 octets (120 octets)
 * - property: 242 octets (128 octets)
 * - observer: 32 octets (8 octets)
 * - actor   : 48 octets (8 octets)
 */
class PVHive
{
private:
	typedef std::set<PVActorBase*> actors_t;
	typedef std::list<PVObserverBase*> observers_t;
	typedef std::set<void*> properties_t;

	struct observable_t
	{
		actors_t     actors;
		observers_t  observers;
		properties_t properties;
	};

	typedef tbb::concurrent_hash_map<void*, observable_t > observables_t;

private:
	template<typename T>
	friend void __impl::hive_deleter(T *ptr);
	friend class PVActorBase;
	template<typename T>
	friend class PVActor;

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
	void register_object(PVCore::pv_shared_ptr<T>& object)
	{
		observables_t::accessor acc;

		_observables.insert(acc, (void*) object.get());
		object.set_deleter(&__impl::hive_deleter<T>);
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
	void register_object(PVCore::pv_shared_ptr<T>& object, F const &prop_get)
	{
		auto &property = prop_get(*object);

		observables_t::accessor acc;

		// adding property's entry
		_observables.insert(acc, (void*) &property);

		// create/get object's entry
		_observables.insert(acc, (void*) object.get());
		acc->second.properties.insert((void*) &property);
		object.set_deleter(&__impl::hive_deleter<T>);
	}

	/**
	 * Register an actor for an ob,ect
	 *
	 * @param object the managed object
	 * @param actor the actor
	 */
	template <class T>
	void register_actor(PVCore::pv_shared_ptr<T>& object, PVActorBase& actor)
	{
		// an actor must be set for only one object
		assert(actor.get_object() == nullptr);

		observables_t::accessor acc;

		// create/get object's entry
		_observables.insert(acc, (void*) object.get());

		// create/get object's entry
		acc->second.actors.insert(&actor);
		actor.set_object((void*) object.get());
		object.set_deleter(&__impl::hive_deleter<T>);
	}

	/**
	 * Helper method easily create and register an actor for an object
	 *
	 * @param object the observed object
	 * @return the actor
	 */
	template <class T>
	PVActor<T>* register_actor(PVCore::pv_shared_ptr<T>& object)
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
	void register_observer(PVCore::pv_shared_ptr<T>& object, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer.get_object() == nullptr);

		observables_t::accessor acc;

		// create/get object's entry
		_observables.insert(acc, (void*) object.get());

		auto res = std::find(acc->second.observers.begin(),
		                     acc->second.observers.end(), &observer);
		if (res == acc->second.observers.end()) {
			acc->second.observers.push_back(&observer);
			observer.set_object((void*) object.get());
			object.set_deleter(&__impl::hive_deleter<T>);
		}
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
	void register_observer(PVCore::pv_shared_ptr<T>& object, F const &prop_get, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer.get_object() == nullptr);

		auto &property = prop_get(*object);

		observables_t::accessor acc;

		// create/get property's entry
		_observables.insert(acc, (void*) &property);

		auto res = std::find(acc->second.observers.begin(),
		                     acc->second.observers.end(), &observer);
		if (res == acc->second.observers.end()) {
			// adding observer
			acc->second.observers.push_back(&observer);
			observer.set_object((void*) &property);

			// adding property
			// create/get object's entry
			_observables.insert(acc, (void*) object.get());
			acc->second.properties.insert((void*) &property);
			object.set_deleter(&__impl::hive_deleter<T>);
		}
	}

public:
	/**
	 * Unregister an actor an notify all dependent observers
	 * that they must stop observing
	 *
	 * @param actor the actor
	 */
	bool unregister_actor(PVActorBase& actor);

	/**
	 * Unregister an observer
	 *
	 * @param observer the observer
	 */
	bool unregister_observer(PVObserverBase& observer);

public:
	/**
	 * Write on std::cout a structured view of PVHive content
	 */
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

	/**
	 * Returns memory usage
	 */
	size_t memory() const
	{
		size_t s = sizeof(observables_t);

		// memory used by observables_t's entries
		s += _observables.size() * sizeof (observables_t::value_type);

		// memory used by entries values
		for (observables_t::const_iterator it = _observables.begin();
		     it != _observables.end(); ++it) {
			s += it->second.actors.size() * sizeof(actors_t::value_type);
			s += it->second.observers.size() * sizeof(observers_t::value_type);
			s += it->second.properties.size() * sizeof(properties_t::value_type);
		}

		return s;
	}

protected:
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

	/**
	 * Unregister an object
	 *
	 * @param object the object
	 */
	void unregister_object(void *object);

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
		if (object == nullptr) {
			return;
		}

		observables_t::accessor acc;

		if (_observables.find(acc, (void*) object)) {
			(object->*f)(params...);
			acc.release();
			do_refresh_observers((void*) object);
		}
	}

	/**
	 * Propagate refresh event to all observers
	 *
	 * @param object the object which has been modified
	 */
	void do_refresh_observers(void *object);

private:
	// private to secure the singleton
	PVHive() {}
	~PVHive() {}
	PVHive(const PVHive&) {}
	PVHive &operator=(const PVHive&) { return *this; }

private:
	static PVHive *_hive;

	observables_t _observables;
};

/**
 * @attention any specialization of template method ::call_object() *must* be
 * declared in the namespace PVHive (you can use the macros
 * PVHIVE_CALL_OBJECT_BLOCK_BEGIN() and PVHIVE_CALL_OBJECT_BLOCK_END()).
 */

#define PVHIVE_CALL_OBJECT_BLOCK_BEGIN() namespace PVHive {
#define PVHIVE_CALL_OBJECT_BLOCK_END() }

namespace __impl
{

// definition of hive_deleter
template <typename T>
inline void hive_deleter(T *ptr)
{
	PVHive::get().unregister_object((void*) ptr);
	delete ptr;
}

}

}

#endif // LIBPVHIVE_PVHIVE_H
