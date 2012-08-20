/**
 * \file PVActorBase.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LIVPVHIVE_PVACTORBASE_H
#define LIVPVHIVE_PVACTORBASE_H

namespace PVHive
{

class PVHive;

/**
 * @class PVActorBase
 *
 * A non template class to use as a base in the PVHive.
 */
class PVActorBase
{
public:
	friend class PVHive;

public:
	PVActorBase() : _object(nullptr)
	{}

	virtual ~PVActorBase();

protected:
	void* get_object() const
	{
		return _object;
	}

	void* get_registered_object() const
	{
		return _registered_object;
	}

	void set_object(void *object, void* registered_object)
	{
		_object = object;
		_registered_object = registered_object;
	}

private:
	void *_object;
	void *_registered_object;
};

}

#endif /* LIVPVHIVE_PVACTORBASE_H */
