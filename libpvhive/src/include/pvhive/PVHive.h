
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <set>
#include <unordered_map>
#include <type_traits>

#include <boost/thread.hpp>

#include <pvhive/PVObserver.h>
#include <pvhive/PVActorBase.h>

namespace PVHive
{

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
	 * Register an actor for an obect
	 *
	 * @param p the managed object
	 * @param actor the actor
	 */
	template <class T>
	void register_actor(T& object, PVActorBase& actor)
	{
		// an actor must be set for only one object
		assert(actor._object == nullptr);

		actor._object = (void*) &object;
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

		observer._object = (void*) &object;

		write_lock_t write_lock(_observables_lock);
		_observables[(void*) &object].first.insert(&observer);
	}

	/**
	 * Register an observer for a member variable of an object
	 *
	 * @param object the parent object
	 * @prop_get: a function returning a reference on object's property
	 * @param observer the observer
	 *
	 * @attention using a method as pro_get will not compile.
	 */
	template <class T, class F>
	void register_observer(T const& object, F const &prop_get, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer._object == nullptr);

		auto &property = prop_get(object);
		write_lock_t write_lock(_observables_lock);
		// inserting observer
		_observables[(void*) &property].first.insert(&observer);
		// inserting property
		_observables[(void*) &object].second.insert((void*) &property);
		observer._object = (void*) &property;
	}

	/**
	 * Unregister an actor an notify all dependent observers
	 * that they must stop observing
	 *
	 * @param actor the actor
	 */
	void unregister_actor(PVActorBase& actor)
	{
		// the actor must have a valid object
		assert(actor._object != nullptr);

		actor._object = nullptr;
	}

	/**
	 * Unregister an observer
	 *
	 * @param observer the observer
	 */
	void unregister_observer(PVObserverBase& observer);

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

	typedef std::set<PVObserverBase*> observers_t;
	typedef std::set<void*> properties_t;
	typedef std::unordered_map<void*, std::pair<observers_t, properties_t> > observables_t;
	observables_t _observables;

	// thread safety
	typedef boost::unique_lock<boost::shared_mutex> write_lock_t;
	typedef boost::shared_lock<boost::shared_mutex> read_lock_t;
	boost::shared_mutex _observables_lock;
};

}

#endif // LIBPVHIVE_PVHIVE_H
