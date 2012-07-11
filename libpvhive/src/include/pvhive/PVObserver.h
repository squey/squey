
#ifndef LIBPVHIVE_PVOBSERVER_H
#define LIBPVHIVE_PVOBSERVER_H

#include <cassert>

namespace PVHive
{

class PVHive;

/**
 * @class PVObserverBase
 *
 * A non template class to use as a base in the PVHive.
 */
class PVObserverBase
{
public:
	friend class PVHive;

public:
	PVObserverBase() : _object(nullptr) {}
	virtual ~PVObserverBase();

protected:
	/**
	 * Action to do when the "refresh" event occurs.
	 */
	virtual void refresh() = 0;

	/**
	 * Action to do when the "about_to_be_deleted" event occurs.
	 */
	virtual void about_to_be_deleted() = 0;

protected:
	void* _object;
};

/**
 * @class PVObserver
 *
 * A template class to specify observers on a given type/class.
 *
 * All subclasses must implements PVObserverBase::refresh() and
 * PVObserverBase::about_to_be_deleted().
 */
template <class T>
class PVObserver : public PVObserverBase
{
public:
	friend class PVHive;

	/**
	 * Getter for observed object (with the right type).
	 *
	 * @return the address of the observed object
	 */
	T const* get_object() const
	{
		return const_cast<T*>((T*)_object);
	}
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
