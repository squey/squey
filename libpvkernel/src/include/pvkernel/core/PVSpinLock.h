/**
 * \file PVSpinLock.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_SPINLOCK_H
#define PVCORE_SPINLOCK_H

#include <atomic>

namespace PVCore
{

struct pv_spin_lock_t : public std::atomic_flag
{
	pv_spin_lock_t()
	{
		atomic_flag_clear(this);
	}
};

class pv_spin_lock_guard_t
{
public:
	pv_spin_lock_guard_t(pv_spin_lock_t &sl) : _sl(sl)
	{
		while(_sl.test_and_set(std::memory_order_acquire));
	}

	~pv_spin_lock_guard_t()
	{
		_sl.clear(std::memory_order_release);
	}

	pv_spin_lock_guard_t(const pv_spin_lock_guard_t&) = delete;
	pv_spin_lock_guard_t& operator=(const pv_spin_lock_guard_t&) = delete;

private:
	pv_spin_lock_t &_sl;
};

}

#endif // PVCORE_SPINLOCK_H
