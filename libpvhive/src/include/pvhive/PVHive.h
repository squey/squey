
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
	typedef std::multimap<void*, PVActorBase*> actors_t;
	typedef std::multimap<void*, PVObserverBase*> observers_t;

public:
	static PVHive &get()
	{
		if (_hive == nullptr) {
			_hive = new PVHive;
		}
		return *_hive;
	}

public:
	template <class T>
	void register_actor(T& p, PVActor<T>& actor)
	{
		_actors.insert(std::make_pair((void*) &p, (PVActorBase*) &actor));
		actor._object = &p;
	}

	template <class T>
	void register_observer(T const& p, PVObserver<T>& observer)
	{
		_observers.insert(std::make_pair((void*) &p, (PVObserverBase*) &observer));
		observer.set_object(&p);
	}

public:
	template <typename T, typename F, F f, typename... Ttypes>
	void call_object(T* obj, Ttypes... params)
	{
		call_object_default<T, F, f>(obj, params...);
	}

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
	template <typename T, typename F, F f, typename... Ttypes>
	void call_object_default(T* obj, Ttypes... params)
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
