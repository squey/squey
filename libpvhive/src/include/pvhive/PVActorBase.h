#ifndef LIVPVHIVE_PVACTORBASE_H
#define LIVPVHIVE_PVACTORBASE_H

#include <pvkernel/core/PVSpinLock.h>

namespace PVHive
{
class PVHive;

class PVActorBase
{
public:
	friend class PVHive;
public:
	PVActorBase() : _object(nullptr)
	{}

	virtual ~PVActorBase();

protected:
	void *get_object() const
	{
		return _object;
	}

private:
	void set_object(void *object)
	{
		PVCore::pv_spin_lock_guard_t slg(_spinlock);
		_object = object;
	}

protected:
	PVCore::pv_spin_lock_t _spinlock;

private:
	void *_object;
};

}

#endif /* LIVPVHIVE_PVACTORBASE_H */
