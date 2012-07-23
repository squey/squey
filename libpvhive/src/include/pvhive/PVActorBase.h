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

class PVActorBase
{
public:
	friend class PVHive;
public:
	PVActorBase() : _object(nullptr) {}
	virtual ~PVActorBase();

protected:
	void* _object;
};

}

#endif /* LIVPVHIVE_PVACTORBASE_H */
