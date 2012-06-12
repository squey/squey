
#ifndef LIBPVHIVE_PVHIVE_H
#define LIBPVHIVE_PVHIVE_H

#include <map>

namespace PVHive
{

class PVHive
{
public:
	static PVHive &get()
	{
		if (_hive == nullptr) {
			_hive = new PVHive;
		}
		return *_hive;
	}

public:
	template <typename T, typename F, F f, typename... Ttypes>
	void call_object(T* obj, Ttypes... params)
	{
		call_object_default<T, F, f>(obj, params...);
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
};


}

#endif // LIBPVHIVE_PVHIVE_H
