
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <set>
#include <unordered_map>

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
 *  - the hive which is the keeper of list of action-reaction; It is the
 *    "entity" which do the action
 *  - the actor tell the hive on which object it will do changes
 *  - the observer tell the hive for which object it want update notification
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
	 * Register an actor for an address
	 *
	 * @param p the managed address
	 * @param actor the actor
	 */
	template <class T>
	void register_actor(T& object, PVActorBase& actor)
	{
		// an actor must be set for only one object
		assert(actor._object == nullptr);

		actor._object = (void*) &object;

		boost::lock_guard<boost::mutex> lock(_actors_mutex);
		_actors[(void*) &object].insert(&actor);
	}

	/**
	 * Helper method easily create and register an actor for an address
	 *
	 * @param object the observed address
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
	 * Register an observer for an address
	 *
	 * @param p the observed address
	 * @param observer the observer
	 */
	template <class T>
	void register_observer(T const& object, PVObserverBase& observer)
	{
		// an observer must be set for only one object
		assert(observer._object == nullptr);

		observer._object = (void*) &object;

		write_lock_t write_lock(_observers_lock);
		_observers[(void*) &object].insert(&observer);
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

	/**
	 * Unregister an object
	 *
	 * @param object the object
	 */
	template <typename T>
	void unregister_object(T const &object)
	{
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
	 * @param object the managed address
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
	 * Tell all observers of an address that a change has occurred
	 *
	 * @param object the observed address
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
	 * @param object the managed address
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
	~PVHive();
	PVHive(const PVHive&) {}
	PVHive &operator=(const PVHive&) { return *this; }

private:
	static PVHive *_hive;

	typedef std::unordered_map<void*, std::set<PVActorBase*> > actors_t;
	typedef std::unordered_map<void*, std::set<PVObserverBase*> > observers_t;
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
