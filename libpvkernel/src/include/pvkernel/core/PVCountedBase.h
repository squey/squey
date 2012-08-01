#ifndef PVCOUNTEDBASE_H_
#define PVCOUNTEDBASE_H_

#include <boost/checked_delete.hpp>

namespace PVCore
{

template <typename T>
class PVCountedBase
{
public:
	typedef T type;
	typedef T*  pointer;
	typedef  void(*deleter)(pointer);

public:

    PVCountedBase(deleter d = nullptr) : _deleter(d)
    {
    	_use_count = 1;
    	_weak_count = 1;
    }

    virtual ~PVCountedBase()
    {
    }

    virtual void destroy()
    {
        delete this;
    }

    virtual void dispose(pointer p)
    {
    	if (_deleter != nullptr) {
			(_deleter)(p);
		}
		else {
			delete p;
			//boost::checked_delete(p);
		}
    }

    long add_ref_copy()
	{
    	return _use_count.fetch_and_increment();
	}

    long add_ref_lock()
	{
		return _use_count.fetch_and_increment() != 0;
	}

	void release(pointer p)
	{
		if(_use_count.fetch_and_decrement() == 1)
		{
			dispose(p);
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

	inline void set_deleter(deleter d = nullptr)
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		_deleter = d;
	}

	inline deleter get_deleter() const
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		return _deleter;
	}

private:
	PVCountedBase<T>(PVCountedBase<T> const & );
	PVCountedBase<T> & operator= (PVCountedBase<T> const &);

	mutable tbb::atomic<long> _use_count;
	mutable tbb::atomic<long> _weak_count;

	deleter _deleter;
	mutable pv_spin_lock_t _spin_lock;
};

}

#endif /* PVCOUNTEDBASE_H_ */
