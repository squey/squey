/**
 * \file PVObserver.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

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

public:
	PVObserverBase() : _object(nullptr) {}
	virtual ~PVObserverBase();

protected:
	virtual void refresh() = 0;
	virtual void about_to_be_deleted() = 0;

protected:
	void* _object;
};

template <class T>
class PVObserver : public PVObserverBase
{
public:
	friend class PVHive;

	/**
	 * @return the address of the observed object
	 */
	T const* get_object() const
	{
		return const_cast<T*>((T*)_object);
	}
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
