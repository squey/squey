/**
 * \file PVObserver.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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
	 * Action to do before the "refresh" event occurs.
	 */
	virtual void about_to_be_refreshed() {};

	/**
	 * Action to do when the "refresh" event occurs.
	 */
	virtual void refresh() {};

	/**
	 * Action to do when the "about_to_be_deleted" event occurs.
	 */
	virtual void about_to_be_deleted() {};

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
		assert(PVObserverBase::get_object());
		return const_cast<T*>(reinterpret_cast<T*>(PVObserverBase::get_object()));
	}

	T* get_object()
	{
		assert(PVObserverBase::get_object());
		return reinterpret_cast<T*>(PVObserverBase::get_object());
	}
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
