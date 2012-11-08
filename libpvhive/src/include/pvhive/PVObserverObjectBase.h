/**
 * \file PVObserverObjectBase.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVHIVE_PVOBSERVERBASE_H
#define PVHIVE_PVOBSERVERBASE_H

#include <tbb/atomic.h>

namespace PVHive {

class PVHive;

/**
 * @class PVObserverBase
 *
 * A non template class to use as a base in the PVHive.
 */
class PVObserverObjectBase
{
public:
	friend class PVHive;

public:
	PVObserverObjectBase():
		_object(nullptr),
		_registered_object(nullptr),
		_accept_recursive_refreshes(false)
	{
		_object_about_to_be_unregistered = false;
	}
	virtual ~PVObserverObjectBase() {}

public:
	inline bool accept_recursive_refreshes() const { return _accept_recursive_refreshes; }
	void set_accept_recursive_refreshes(bool accept) { _accept_recursive_refreshes = accept; }

protected:
	void *get_object() const
	{
		return _object;
	}

	void set_object(void *object, void* registered_object)
	{
		_object = object;
		_registered_object = registered_object;
	}


protected:
	void object_about_to_be_unregistered() { _object_about_to_be_unregistered = true; }

protected:
	bool is_object_about_to_be_unregistered() const { return _object_about_to_be_unregistered; }

protected:
	void* _object;
	void* _registered_object;

	// Deadlock mlay occur if an observer of an object is deleted during an about_to_be_deleted operation of that object
	tbb::atomic<bool> _object_about_to_be_unregistered;
	bool _accept_recursive_refreshes;
};

}

#endif
