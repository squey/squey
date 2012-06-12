
#ifndef LIBPVHIVE_PVOBSERVER_H
#define LIBPVHIVE_PVOBSERVER_H

#include <cassert>

namespace PVHive
{

class PVHive;

class PVObserverBase
{
public:
	friend class PVHive;
protected:
	friend class PVHive;

	virtual void refresh() = 0;
	virtual void about_to_be_deleted() = 0;
};

template <class T>
class PVObserver : public PVObserverBase
{
public:
	friend class PVHive;

	PVObserver()
	{
		_object = nullptr;
	}

protected:

	/**
	 * @return the address of the observed object
	 */
	T const* get_object() const
	{
		return _object;
	}

protected:
	T const* _object;
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
