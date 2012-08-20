/**
 * \file PVObserverObjectBase.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVHIVE_PVOBSERVERBASE_H
#define PVHIVE_PVOBSERVERBASE_H

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
	PVObserverObjectBase() : _object(nullptr), _registered_object(nullptr) {}
	virtual ~PVObserverObjectBase() {}

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
	void* _object;
	void* _registered_object;
};

}

#endif
