/**
 * \file PVCountedBase.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVCOUNTEDBASE_H_
#define PVCOUNTEDBASE_H_

#include <typeinfo>
#include <tbb/atomic.h>
#include <boost/checked_delete.hpp>
#include <pvkernel/core/PVSpinLock.h>

namespace PVCore
{

class PVCountedBase
{
public:
    PVCountedBase()
    {
    	_use_count = 1;
    	_weak_count = 1;
    }

    virtual ~PVCountedBase()
    {
    }

     void destroy()
    {
        delete this;
    }

    virtual void dispose() = 0;

    long add_ref_copy()
	{
    	return _use_count.fetch_and_increment();
	}

    long add_ref_lock()
	{
		return _use_count.fetch_and_increment() != 0;
	}

	void release()
	{
		if(_use_count.fetch_and_decrement() == 1)
		{
			dispose();
			weak_release();
		}
	}

	long weak_add_ref()
    {
    	return _weak_count.fetch_and_increment();
    }

    void weak_release()
    {
        if(_weak_count.fetch_and_decrement() == 1)
        {
            destroy();
        }
    }

    long use_count() const
    {
    	return _use_count.fetch_and_add((long)0);
    }

	virtual void* get() = 0;
	virtual void set(void* p) = 0;

	virtual void set_deleter(void* d = nullptr) = 0;
	virtual void* get_deleter() const = 0;

private:
	PVCountedBase(PVCountedBase const & );
	PVCountedBase & operator= (PVCountedBase const &);

	mutable tbb::atomic<long> _use_count;
	mutable tbb::atomic<long> _weak_count;
};

namespace __impl
{
	template <typename P, typename D>
	class PVCountedBasePD : public PVCountedBase
	{
	public:
		PVCountedBasePD(P p, D & d) : PVCountedBase(), _px(p), _deleter(d)
		{
		}

		PVCountedBasePD(P p) : PVCountedBase(), _px(p), _deleter(nullptr)
		{
		}

		virtual ~PVCountedBasePD()
		{
		}

		virtual void dispose()
		{
			if (_deleter) {
				_deleter(_px);
			}
			else {
				boost::checked_delete(_px);
			}
		}

		virtual void* get() { return _px; }
		virtual void set(void* p) { _px = (P) p; }

		virtual void set_deleter(void* d = nullptr)
		{
			pv_spin_lock_guard_t guard(_spin_lock);
			_deleter = (D) d;
		}

		virtual void* get_deleter() const
		{
			pv_spin_lock_guard_t guard(_spin_lock);
			return (void*) _deleter;
		}

	private:
		P _px;
		D _deleter;

		mutable pv_spin_lock_t _spin_lock;

		PVCountedBasePD(PVCountedBasePD const & );
		PVCountedBasePD & operator= (PVCountedBasePD const & );

		typedef PVCountedBasePD<P, D> this_type;
	};
}

}

#endif /* PVCOUNTEDBASE_H_ */
