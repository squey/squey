
#ifndef LIBPVHIVE_PVOBSERVER_H
#define LIBPVHIVE_PVOBSERVER_H

#include <cassert>
#include <pvhive/PVObserverObjectBase.h>

namespace PVHive
{

class PVHive;

class PVObserverBase: public PVObserverObjectBase
{
	friend class PVHive;

public:
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
		return const_cast<T*>((T*)PVObserverBase::get_object());
	}
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
